import os
import tempfile
from pathlib import Path

import streamlit as st

# ---------- Portfolio UI styling ----------
st.set_page_config(
    page_title="Agentic RAG | Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 2.6rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
    }
</style>
""", unsafe_allow_html=True)


from src.db import get_connection, initialize_database
from src.ingest import ingest_file
from src.agent import ask_agent
from src.evaluation import run_evaluation, get_evaluation_history


st.set_page_config(
    page_title="Company Knowledge AI",
    page_icon="📚",
    layout="wide",
)


st.title("🧠 Agentic RAG Knowledge Assistant")
st.caption(
    "Ask questions about your documents • "
    "Evidence-based retrieval • Local LLM • Evaluation dashboard"
)

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
    st.caption(
        "Measure answer correctness, retrieval performance, "
        "and evidence-based abstention across repeatable test runs."
    )

    st.info(
        "Run the four-test evaluation suite against your indexed "
        "sample HR policy. Results depend on the documents available "
        "in your database."
    )

    # Run evaluation
    if st.button(
        "▶ Run Evaluation",
        type="primary",
        key="run_eval",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Evaluating answers, retrieval, and evidence sufficiency..."
            ):
                evaluation = run_evaluation()

            st.session_state["latest_evaluation"] = evaluation
            st.success(
                f"Evaluation run #{evaluation['run_id']} completed."
            )

        except Exception as error:
            st.error(f"Evaluation failed: {error}")

    # Latest evaluation results
    if "latest_evaluation" in st.session_state:
        evaluation = st.session_state["latest_evaluation"]

        st.divider()
        st.markdown("### Latest Evaluation Results")

        st.caption(f"Run ID: {evaluation['run_id']}")

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "Answer Accuracy",
                f"{evaluation['answer_accuracy']:.0%}",
                help="Share of test questions marked correct.",
            )

        with metric2:
            st.metric(
                "Retrieval Hit Rate",
                f"{evaluation['retrieval_hit_rate']:.0%}",
                help="Share of tests where relevant passages were retrieved.",
            )

        with metric3:
            st.metric(
                "Abstention Accuracy",
                f"{evaluation['abstention_accuracy']:.0%}",
                help="Performance on tests involving whether the agent should abstain.",
            )

        with metric4:
            st.metric(
                "Tests Passed",
                f"{evaluation['passed_tests']}/{evaluation['total_tests']}",
                help="Tests passing the evaluator's pass criteria.",
            )

        # Overall pass summary
        total_tests = evaluation["total_tests"]
        passed_tests = evaluation["passed_tests"]

        if total_tests > 0:
            pass_rate = passed_tests / total_tests
            st.progress(
                pass_rate,
                text=f"Overall test pass rate: {pass_rate:.0%}",
            )

        st.markdown("### Test Case Results")

        display_results = []

        for test in evaluation["results"]:
            display_results.append({
                "Test ID": test["test_id"],
                "Question": test["question"],
                "Answer": (
                    "✅ PASS" if test["answer_correct"] else "❌ FAIL"
                ),
                "Retrieval": (
                    "✅ HIT" if test["retrieval_hit"] else "❌ MISS"
                ),
                "Evidence": (
                    "Sufficient" if test["evidence_sufficient"]
                    else "Insufficient"
                ),
                "Attempts": test["attempts"],
            })

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Inspect Individual Answers")

        for index, test in enumerate(evaluation["results"]):
            label = f"{test['test_id']} — {test['question']}"

            with st.expander(label):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Answer correctness**")
                    if test["answer_correct"]:
                        st.success("Passed")
                    else:
                        st.error("Failed")

                with col2:
                    st.markdown("**Evidence assessment**")
                    if test["evidence_sufficient"]:
                        st.success("Sufficient evidence")
                    else:
                        st.warning("Insufficient evidence")

                st.markdown("**Generated answer**")
                st.write(test["answer"])

                st.caption(
                    f"Retrieval attempts: {test['attempts']} · "
                    f"Retrieval hit: "
                    f"{'Yes' if test['retrieval_hit'] else 'No'}"
                )

    # Historical results
    st.divider()
    st.markdown("### Evaluation History")
    st.caption(
        "Compare saved evaluation runs to observe changes "
        "in answer quality and retrieval performance."
    )

    try:
        history = get_evaluation_history()

        if history:
            history_display = [
                {
                    "Run ID": row["Run ID"],
                    "Started At": str(row["Started at"]),
                    "Tests": row["Total tests"],
                    "Passed": row["Passed tests"],
                    "Answer Accuracy": f"{row['Answer accuracy']:.0%}",
                    "Retrieval Hit Rate": f"{row['Retrieval hit rate']:.0%}",
                    "Abstention Accuracy": (
                        f"{row['Abstention accuracy']:.0%}"
                    ),
                }
                for row in history
            ]

            st.dataframe(
                history_display,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Performance Across Evaluation Runs")

            # Plot the oldest run first.
            chart_rows = sorted(
                history,
                key=lambda row: row["Run ID"],
            )

            chart_data = [
                {
                    "Run": str(row["Run ID"]),
                    "Answer Accuracy": row["Answer accuracy"] * 100,
                    "Retrieval Hit Rate": row["Retrieval hit rate"] * 100,
                    "Abstention Accuracy": (
                        row["Abstention accuracy"] * 100
                    ),
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
            st.info(
                "No evaluation runs saved yet. Run the test suite "
                "to create your first evaluation record."
            )

    except Exception as error:
        st.error(f"Could not load evaluation history: {error}")

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