
from datetime import datetime

from src.agent import ask_agent
from src.db import get_connection


TEST_CASES = [
    {
        "id": "annual_leave",
        "question": "How many annual leave days do full-time employees receive?",
        "expected_keywords": ["20 days"],
        "retrieval_keywords": ["annual leave", "20 days"],
        "should_abstain": False,
    },
    {
        "id": "sick_leave",
        "question": "How many sick leave days do employees receive?",
        "expected_keywords": ["10 days"],
        "retrieval_keywords": ["sick leave", "10 days"],
        "should_abstain": False,
    },
    {
        "id": "remote_work",
        "question": "How many days per week can employees work remotely?",
        "expected_keywords": ["two days", "2 days"],
        "retrieval_keywords": ["remotely", "two days per week"],
        "should_abstain": False,
    },
    {
        "id": "parental_leave",
        "question": "What is the duration of parental leave?",
        "expected_keywords": [],
        "retrieval_keywords": ["parental leave"],
        "should_abstain": True,
    },
]


ABSTENTION_SIGNALS = [
    "not enough information",
    "does not provide enough",
    "doesn't provide enough",
    "couldn't find",
    "cannot determine",
    "can't determine",
    "not specified",
    "does not specify",
    "doesn't specify",
    "not stated",
    "not mentioned",
    "insufficient information",
    "i don't know",
    "i cannot confirm",
    "i can't confirm",
]


def ensure_evaluation_tables():
    """Create evaluation tables and safely add new metric columns."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id BIGSERIAL PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                total_tests INTEGER NOT NULL DEFAULT 0,
                passed_tests INTEGER NOT NULL DEFAULT 0,
                answer_accuracy DOUBLE PRECISION NOT NULL DEFAULT 0,
                retrieval_hit_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
                abstention_accuracy DOUBLE PRECISION NOT NULL DEFAULT 0
            )
        """)

        conn.execute("""
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS abstention_accuracy
            DOUBLE PRECISION NOT NULL DEFAULT 0
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id BIGSERIAL PRIMARY KEY,
                run_id BIGINT NOT NULL REFERENCES evaluation_runs(id)
                    ON DELETE CASCADE,
                test_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                answer_correct BOOLEAN NOT NULL,
                retrieval_hit BOOLEAN NOT NULL,
                evidence_sufficient BOOLEAN NOT NULL,
                attempts INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        conn.execute("""
            ALTER TABLE evaluation_results
            ADD COLUMN IF NOT EXISTS should_abstain
            BOOLEAN NOT NULL DEFAULT FALSE
        """)

        conn.execute("""
            ALTER TABLE evaluation_results
            ADD COLUMN IF NOT EXISTS abstention_correct
            BOOLEAN NOT NULL DEFAULT FALSE
        """)


def evaluate_one(test_case):
    """Evaluate one answer and the evidence retrieved for it."""
    result = ask_agent(test_case["question"])

    answer = str(result["answer"]).lower()
    sources = result["sources"]

    source_text = " ".join(
        str(source.get("content", "")).lower()
        for source in sources
    )

    # For factual questions, any expected answer variant is accepted.
    expected_keywords = test_case["expected_keywords"]

    if test_case["should_abstain"]:
        answer_correct = any(
            signal in answer for signal in ABSTENTION_SIGNALS
        )
    else:
        answer_correct = any(
            keyword in answer for keyword in expected_keywords
        )

    # Require every listed evidence phrase, not just one of them.
    retrieval_keywords = test_case["retrieval_keywords"]

    retrieval_hit = bool(retrieval_keywords) and all(
        keyword in source_text
        for keyword in retrieval_keywords
    )

    # Only abstention-required tests contribute to abstention accuracy.
    should_abstain = test_case["should_abstain"]
    abstention_correct = (
        answer_correct if should_abstain else False
    )

    return {
        "test_id": test_case["id"],
        "question": test_case["question"],
        "answer": result["answer"],
        "answer_correct": answer_correct,
        "retrieval_hit": retrieval_hit,
        "evidence_sufficient": result["evidence_sufficient"],
        "attempts": result["attempts"],
        "should_abstain": should_abstain,
        "abstention_correct": abstention_correct,
    }


def run_evaluation():
    """Run all test cases, calculate metrics, and save results."""
    ensure_evaluation_tables()

    results = [
        evaluate_one(test_case)
        for test_case in TEST_CASES
    ]

    total = len(results)

    passed = sum(
        row["answer_correct"] and row["retrieval_hit"]
        for row in results
    )

    answer_accuracy = (
        sum(row["answer_correct"] for row in results) / total
        if total else 0
    )

    retrieval_hit_rate = (
        sum(row["retrieval_hit"] for row in results) / total
        if total else 0
    )

    abstention_tests = [
        row for row in results if row["should_abstain"]
    ]

    abstention_accuracy = (
        sum(row["abstention_correct"] for row in abstention_tests)
        / len(abstention_tests)
        if abstention_tests else 0
    )

    with get_connection() as conn:
        run_row = conn.execute(
            """
            INSERT INTO evaluation_runs (
                total_tests,
                passed_tests,
                answer_accuracy,
                retrieval_hit_rate,
                abstention_accuracy
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                total,
                passed,
                answer_accuracy,
                retrieval_hit_rate,
                abstention_accuracy,
            ),
        ).fetchone()

        run_id = run_row[0]

        for row in results:
            conn.execute(
                """
                INSERT INTO evaluation_results (
                    run_id,
                    test_id,
                    question,
                    answer,
                    answer_correct,
                    retrieval_hit,
                    evidence_sufficient,
                    attempts,
                    should_abstain,
                    abstention_correct
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    row["test_id"],
                    row["question"],
                    row["answer"],
                    row["answer_correct"],
                    row["retrieval_hit"],
                    row["evidence_sufficient"],
                    row["attempts"],
                    row["should_abstain"],
                    row["abstention_correct"],
                ),
            )

    return {
        "run_id": run_id,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total,
        "passed_tests": passed,
        "answer_accuracy": answer_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "abstention_accuracy": abstention_accuracy,
        "results": results,
    }


def get_evaluation_history():
    """Retrieve the 20 most recent saved evaluation runs."""
    ensure_evaluation_tables()

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                id,
                started_at,
                total_tests,
                passed_tests,
                answer_accuracy,
                retrieval_hit_rate,
                abstention_accuracy
            FROM evaluation_runs
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

    return [
        {
            "Run ID": row[0],
            "Started at": row[1],
            "Total tests": row[2],
            "Passed tests": row[3],
            "Answer accuracy": row[4],
            "Retrieval hit rate": row[5],
            "Abstention accuracy": row[6],
        }
        for row in rows
    ]
