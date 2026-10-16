
from pathlib import Path

import pymupdf
from docx import Document
from sentence_transformers import SentenceTransformer

from src.db import get_connection


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    """Load the local embedding model."""
    model = SentenceTransformer(MODEL_NAME)

    if model.get_sentence_embedding_dimension() != 384:
        raise ValueError("The embedding model must return 384 dimensions.")

    return model


def extract_pages(file_path):
    """Return a list of (page_number, text) pairs."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            return [
                (page_number, page.get_text())
                for page_number, page in enumerate(pdf, start=1)
            ]

    if suffix == ".docx":
        doc = Document(str(path))
        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )
        return [(None, text)]

    if suffix == ".txt":
        return [(None, path.read_text(encoding="utf-8-sig"))]

    raise ValueError("Supported formats: PDF, DOCX, TXT")


def split_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def ingest_file(file_path, original_filename=None):
    """Extract, embed and store a document."""
    file_path = Path(file_path)
    filename = original_filename or file_path.name

    pages = extract_pages(file_path)
    chunks = []

    for page_number, page_text in pages:
        for chunk in split_text(page_text):
            chunks.append((page_number, chunk))

    if not chunks:
        raise ValueError(
            "No text found. Scanned PDFs may require OCR."
        )

    model = load_model()
    texts = [chunk_text for _, chunk_text in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    with get_connection() as conn:
        # Replace an earlier upload with the same filename.
        existing = conn.execute(
            "SELECT id FROM documents WHERE filename = %s",
            (filename,),
        ).fetchone()

        if existing:
            conn.execute(
                "DELETE FROM documents WHERE id = %s",
                (existing[0],),
            )

        document_id = conn.execute(
            """
            INSERT INTO documents (filename)
            VALUES (%s)
            RETURNING id
            """,
            (filename,),
        ).fetchone()[0]

        for index, ((page_number, content), embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            conn.execute(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, page_number,
                     content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    index,
                    page_number,
                    content,
                    embedding,
                ),
            )

    return len(chunks)
