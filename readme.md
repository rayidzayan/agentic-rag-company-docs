# 🧠 Agentic RAG over Company Documents

An AI-powered document question-answering system built with Retrieval-Augmented Generation (RAG), PostgreSQL with pgvector, LangGraph, and a local Ollama language model.

Upload company documents, ask questions in natural language, inspect retrieved evidence, and evaluate system performance through an interactive dashboard.

## ✨ Features

- **Document ingestion:** Upload PDF, DOCX, and TXT files.
- **Semantic search:** Find relevant document passages using vector embeddings.
- **Agentic retrieval:** Assess evidence and refine queries when the initial retrieval is insufficient.
- **Local LLM inference:** Generate answers using Ollama.
- **Source inspection:** Review retrieved passages to understand the evidence behind answers.
- **Evaluation dashboard:** Measure answer accuracy, retrieval hit rate, and abstention accuracy.
- **Evaluation history:** Compare results across saved test runs.
- **PostgreSQL persistence:** Store document chunks, embeddings, and evaluation records.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Database | PostgreSQL |
| Vector search | pgvector |
| Embeddings | Sentence Transformers |
| Agent orchestration | LangGraph |
| Language model | Ollama |
| Language | Python |

## 🏗️ Architecture

See [`docs/architecture.md`](docs/architecture.md) for the system architecture diagram and component overview.

## ⚙️ Setup

### Prerequisites

Install Python, PostgreSQL, and Ollama. Enable the pgvector extension in your PostgreSQL database and download the local language model configured in `src/agent.py`.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd company-rag
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and update the PostgreSQL settings for your local database.

Never commit your actual `.env` file or credentials.

### 5. Start Ollama

Make sure the model configured in `src/agent.py` is installed and available locally. For example, if the configured model is `llama3.2`:

```powershell
ollama pull llama3.2
```

### 6. Run the application

```powershell
streamlit run app.py
```

## 📊 Evaluation

The evaluation dashboard runs a repeatable test suite against indexed documents. It reports answer accuracy, retrieval hit rate, abstention accuracy, test outcomes, and historical performance.

Evaluation results depend on the documents, test cases, and evaluator criteria. The included test suite is a demonstration rather than a guarantee of production-level accuracy.

## 🔐 Privacy and Security

The application can use a local Ollama model for inference. Document storage and processing depend on the configured environment and libraries. Review your configuration before using confidential company information.

## 🚧 Future Improvements

- Larger and more diverse evaluation datasets.
- Stronger relevance and faithfulness metrics.
- Document-level access control.
- Improved source citations and answer verification.
- Deployment and automated testing.

## 👨‍💻 About

Built as a portfolio project demonstrating document ingestion, vector retrieval, agentic workflows, local language models, and evaluation dashboards.

**Note:** Update this README with screenshots, your repository URL, and any setup details specific to your implementation before publishing.
