import os
import tempfile
from pathlib import Path

import streamlit as st

from src.db import get_connection, initialize_database
from src.ingest import ingest_file
from src.agent import ask_agent
from src.evaluation import run_evaluation, get_evaluation_history


st.set_page_config(
    page_title="Company Knowledge AI",
    page_icon="📚",
    layout="wide",
)

st.title("Company Knowledge AI")
st.caption("Agentic RAG · Local AI · PostgreSQL + pgvector")

# Initialize PostgreSQL tables
initialize_database()


# --------------------------------------------------
# SIDEBAR: DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.header("Document Library")

    uploaded_files = st.file_uploader(
        "Upload company documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one document.")
        else:
            for uploaded in uploaded_files:
                temp_path = None

                try:
                    suffix = Path(uploaded.name).suffix.lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:
                        temp_file.write(uploaded.getvalue())
                        temp_path = temp_file.name

                    with st.spinner(f"Indexing {uploaded.name}..."):
                        count = ingest_file(
                            temp_path,
                            original_filename=uploaded.name,
                        )

                    st.success(
                        f"{uploaded.name}: {count} chunks indexed."
                    )

                except Exception as error:
                    st.error(f"Indexing failed: {error}")

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

tab_qa, tab_eval = st.tabs([
    "💬 Ask Documents",
    "📊 Evaluation Dashboard",
])


# --------------------------------------------------
# TAB 1: ASK QUESTIONS
# --------------------------------------------------

with tab_qa:
    st.subheader("Ask your company documents")

    question = st.text_input(
        "Your question",
        placeholder="How many annual leave days do employees receive?",
        key="question_input",
    )

    if st.button("Ask AI", type="primary", key="ask_ai"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(
                    "Agent is retrieving evidence and generating an answer..."
                ):
                    result = ask_agent(question.strip())

                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()

            except Exception as error:
                st.error(f"Agent failed: {error}")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.markdown("### Question")
        st.write(st.session_state.get("last_question", ""))

        st.markdown("### Answer")
        st.markdown(result["answer"])

        st.markdown("### Evidence status")

        if result["evidence_sufficient"]:
            st.success(
                "The evidence evaluator marked the passages sufficient."
            )
        else:
            st.warning(
                "The evidence evaluator could not confirm sufficient evidence."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Retrieval attempts", result["attempts"])

        with col2:
            st.metric("Passages retrieved", len(result["sources"]))

        with st.expander("Agent execution trace", expanded=True):
            for step in result["trace"]:
                st.write("• " + step)

        with st.expander("Retrieved source passages"):
            for index, source in enumerate(result["sources"], start=1):
                page = source.get("page_number")

                location = (
                    f" · Page {page}"
                    if page is not None
                    else ""
                )

                st.markdown(
                    f"**[{index}] {source['filename']}{location}**"
                )
                st.caption(
                    f"Similarity: {source['similarity']:.3f}"
                )
                st.write(source["content"])
                st.divider()


# --------------------------------------------------
# TAB 2: EVALUATION DASHBOARD
# --------------------------------------------------

with tab_eval:
    st.subheader("RAG Evaluation Dashboard")

    st.write(
        "Run a repeatable test suite to measure answer correctness "
        "and whether the retriever finds relevant passages."
    )

    st.info(
        "The evaluation runs four test questions against your indexed "
        "documents. Make sure your sample HR policy is indexed first."
    )

    if st.button("▶ Run Evaluation", type="primary", key="run_eval"):
        try:
            with st.spinner(
                "Running evaluation tests. This may take a few minutes..."
            ):
                evaluation = run_evaluation()

            st.session_state["latest_evaluation"] = evaluation
            st.success(
                f"Evaluation run #{evaluation['run_id']} completed."
            )

        except Exception as error:
            st.error(f"Evaluation failed: {error}")

    if "latest_evaluation" in st.session_state:
        evaluation = st.session_state["latest_evaluation"]

        st.markdown("### Latest Evaluation Results")

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "Answer Accuracy",
                f"{evaluation['answer_accuracy']:.0%}",
            )

        with metric2:
            st.metric(
                "Retrieval Hit Rate",
                f"{evaluation['retrieval_hit_rate']:.0%}",
            )

        with metric3:
            st.metric(
                "Abstention Accuracy",
                f"{evaluation['abstention_accuracy']:.0%}",
            )

        with metric4:
            st.metric(
                "Tests Passed",
                f"{evaluation['passed_tests']}/{evaluation['total_tests']}",
            )

        st.caption(f"Evaluation Run ID: {evaluation['run_id']}")

        display_results = []

        for test in evaluation["results"]:
            display_results.append({
                "Test": test["test_id"],
                "Question": test["question"],
                "Answer Correct": (
                    "PASS" if test["answer_correct"] else "FAIL"
                ),
                "Retrieval Hit": (
                    "PASS" if test["retrieval_hit"] else "FAIL"
                ),
                "Evidence Sufficient": (
                    "Yes" if test["evidence_sufficient"] else "No"
                ),
                "Attempts": test["attempts"],
            })

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View test answers"):
            for test in evaluation["results"]:
                st.markdown(f"**{test['test_id']}**")
                st.write(test["question"])
                st.write(test["answer"])
                st.divider()

    st.divider()
    st.markdown("### Evaluation History")

    try:
        history = get_evaluation_history()

        if history:
            history_display = [
                {
                    "Run ID": row["Run ID"],
                    "Started At": row["Started at"],
                    "Tests": row["Total tests"],
                    "Passed": row["Passed tests"],
                    "Answer Accuracy": f"{row['Answer accuracy']:.0%}",
                    "Retrieval Hit Rate": f"{row['Retrieval hit rate']:.0%}",
                    "Abstention Accuracy": f"{row['Abstention accuracy']:.0%}",
                }
                for row in history
            ]

            st.dataframe(
                history_display,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Performance Across Evaluation Runs")

            # Display runs from oldest to newest on the chart.
            chart_rows = sorted(
                history,
                key=lambda row: row["Run ID"],
            )

            chart_data = [
                {
                    "Run": str(row["Run ID"]),
                    "Answer Accuracy": row["Answer accuracy"] * 100,
                    "Retrieval Hit Rate": row["Retrieval hit rate"] * 100,
                    "Abstention Accuracy": row["Abstention accuracy"] * 100,
                }
                for row in chart_rows
            ]

            
            st.line_chart(
                chart_data,
                x="Run",
                y=[
                    "Answer Accuracy",
                    "Retrieval Hit Rate",
                    "Abstention Accuracy",
                  ],
                ) 



        else:
            st.info("No evaluation runs saved yet.")

    except Exception as error:
        st.error(f"Could not load evaluation history: {error}")


# --------------------------------------------------
# INDEXED DOCUMENTS
# --------------------------------------------------

st.divider()
st.subheader("Indexed Documents")

try:
    with get_connection() as conn:
        documents = conn.execute("""
            SELECT
                d.filename,
                d.uploaded_at,
                COUNT(c.id) AS chunks
            FROM documents AS d
            LEFT JOIN document_chunks AS c
                ON c.document_id = d.id
            GROUP BY d.id, d.filename, d.uploaded_at
            ORDER BY d.uploaded_at DESC
        """).fetchall()

    if documents:
        st.dataframe(
            [
                {
                    "Filename": row[0],
                    "Uploaded": row[1],
                    "Chunks": row[2],
                }
                for row in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Upload a document to get started.")

except Exception as error:
    st.error(f"Could not load indexed documents: {error}")