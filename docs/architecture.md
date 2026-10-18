
# System Architecture

```mermaid
flowchart TD
    A[PDF / DOCX / TXT Documents] --> B[Document Text Extraction]
    B --> C[Text Chunking]
    C --> D[Sentence Transformers Embeddings]
    D --> E[(PostgreSQL + pgvector)]

    U[User Question] --> F[LangGraph Agent]
    F --> G[Vector Search]
    E --> G
    G --> H{Evidence Sufficient?}
    H -- No --> I[Refine Query and Retrieve Again]
    I --> G
    H -- Yes --> J[Ollama Local LLM]
    J --> K[Answer with Source Passages]

    L[Evaluation Test Cases] --> M[Evaluation Metrics]
    M --> N[(PostgreSQL Evaluation History)]
    N --> O[Streamlit Evaluation Dashboard]
    K --> P[Streamlit Q&A Interface]
```

## Main Components

- **Streamlit:** User interface and evaluation dashboard.
- **PostgreSQL + pgvector:** Stores document chunks and vector embeddings.
- **Sentence Transformers:** Converts text into embeddings for semantic search.
- **LangGraph:** Orchestrates retrieval, evidence assessment, and query refinement.
- **Ollama:** Runs the local language model.
- **Evaluation module:** Tests answer quality, retrieval behavior, and abstention.
