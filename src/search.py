
from sentence_transformers import SentenceTransformer

from src.db import get_connection
from src.ingest import MODEL_NAME


def search_documents(question, top_k=5):
    """Find the most relevant document chunks."""
    question = question.strip()

    if not question:
        return []

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                d.filename,
                c.page_number,
                c.content,
                1 - (c.embedding <=> %s) AS similarity
            FROM document_chunks AS c
            JOIN documents AS d
                ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()

    return [
        {
            "filename": row[0],
            "page_number": row[1],
            "content": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]
