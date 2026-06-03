# Jezikovne tehnologije RAG Server

Idiom translation RAG pipeline with:
- embedding API (HTTP)
- Qdrant vector DB
- web UI + API server (FastAPI)
- optional Gradio chat UI

## Project files

- `config.py` - environment config
- `data_loader.py` - loads and normalizes CSV idiom datasets
- `embedder_api.py` - embedding API client
- `build_index.py` - index build logic + CLI (`--recreate-collection`)
- `train.py` - explicit training command wrapper
- `retriever.py` - top-k retrieval from Qdrant
- `translate.py` - CLI query tool
- `evaluate.py` - simple Hit@1 evaluation
- `web_server.py` - FastAPI backend + browser UI routes
- `run_server.py` - starts FastAPI server via uvicorn
- `web/index.html` - browser UI (chat + train button)
- `chatbot_gui.py` - optional Gradio chat UI
- `run_all.py` - one command: optional index + selected UI

## 1) Setup

Use Python `3.10`-`3.12` (recommended).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Qdrant (example with Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 2) Environment variables

```bash
export EMBEDDING_API_URL="https://your-api/embeddings"
export EMBEDDING_API_KEY="YOUR_BEARER_TOKEN"
export EMBEDDING_API_MODEL="bge-m3"
export EMBEDDING_API_TIMEOUT="120"
export EMBEDDING_BATCH_SIZE="16"
export EMBEDDING_API_RETRIES="2"
export EMBEDDING_API_BACKOFF_SEC="1.5"

export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_COLLECTION="Beees"
export QDRANT_API_KEY=""

export IDIOM_DATA_ROOT="/path/to/folder/with/csv/files"

# optional LLM-like chat API
export CHAT_API_URL=""
export CHAT_API_KEY=""
export CHAT_API_MODEL=""
export CHAT_API_TIMEOUT="120"
export CHAT_TOP_K="5"
export CHAT_HISTORY_TURNS="4"

export RETRIEVAL_MIN_SCORE="0.15"

# Gradio-only options
export GUI_HOST="127.0.0.1"
export GUI_PORT="7860"
```

## 3) Train (build index)

Training in this project means: embed your dataset and index it in Qdrant.

```bash
python train.py
```

Force full retrain (drop old collection and rebuild):

```bash
python train.py --recreate-collection
```

You can also run the original builder directly:

```bash
python build_index.py --recreate-collection
```

## 4) Run UI locally

Run FastAPI server + browser UI:

```bash
python run_server.py --host 0.0.0.0 --port 8000
```

Open:
- `http://localhost:8000/`

The UI includes:
- chat with your indexed idioms
- one-click training (`/api/train`)

## 5) One-command run (index + UI)

Default mode starts FastAPI server UI:

```bash
python run_all.py
```

Options:

```bash
python run_all.py --skip-index
python run_all.py --force-reindex
python run_all.py --ui server --server-host 0.0.0.0 --server-port 8000
python run_all.py --ui gradio
```

## 6) Run on a server (Linux VM / cloud)

Recommended production command:

```bash
uvicorn web_server:app --host 0.0.0.0 --port 8000 --workers 2
```

Then put Nginx/Caddy in front (TLS + reverse proxy).

## 7) CLI usage and evaluation

```bash
python translate.py --query "spill the beans" --top-k 5
python evaluate.py
python evaluate.py --test-csv ./tests/eval.csv
```
