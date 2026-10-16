
# Agentic RAG over Company Documents

A local AI-powered question-answering system that retrieves evidence from company documents and evaluates answer quality.

## Features

- Upload PDF, DOCX, and TXT documents
- Semantic search using PostgreSQL and pgvector
- Local language model inference using Ollama
- Agentic retrieval and query refinement with LangGraph
- Evidence sufficiency checks
- Evaluation dashboard with answer accuracy, retrieval hit rate, and abstention accuracy
- PostgreSQL-backed evaluation history
- Historical performance charts built with Streamlit

## Technology Stack

- Python
- Streamlit
- PostgreSQL
- pgvector
- Sentence Transformers
- Ollama
- LangGraph

## Requirements

- Python version compatible with your installed project dependencies
- PostgreSQL with the pgvector extension
- Ollama with the model configured in `src/agent.py`

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd agentic-rag-company-docs
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use the Python version that works with your installed dependencies.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and update the values for your local PostgreSQL installation.

Never commit your real `.env` file or credentials.

### 5. Prepare PostgreSQL

Create the project database, then enable pgvector in that database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application initializes its required tables when it starts.

### 6. Prepare Ollama

Install Ollama and download the model configured in `src/agent.py`.

For example, if the configured model is `llama3.2`:

```powershell
ollama pull llama3.2
```

Make sure Ollama is running before asking questions.

### 7. Start the application

```powershell
python -m streamlit run app.py
```

## Evaluation

The evaluation dashboard runs a small set of predefined test questions and reports:

- Answer accuracy
- Retrieval hit rate
- Abstention accuracy
- Combined test pass count
- Historical results

These are initial rule-based metrics intended for a demonstration. They are not a substitute for a larger, independently reviewed evaluation dataset.

## Security

- Keep database credentials in `.env`.
- Do not upload confidential company documents.
- Use only documents you have permission to share.
- Review uploaded files before publishing this repository.

## Project Status

Working local prototype. Setup requirements and model configuration may need adjustment for other machines.
