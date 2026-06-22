# NVIDIA AI CEO Strategic Intelligence Agent

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  RSS Feeds   │  │   Reddit     │  │  NewsAPI / IR     │  │
│  │ (5 sources)  │  │ (4 subreddits)│  │  (investor page)  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          └─────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Clean Text  │→ │ Chunk (400)  │→ │ Embed (MiniLM)     │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           ChromaDB (cosine similarity)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────┐                                     │
│  │ Sentiment (RoBERTa)  │                                     │
│  └──────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              INTELLIGENCE ENGINE (RAG)                       │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Opportunities   │  │    Risks     │  │   Trends     │   │
│  │  (query → LLM)   │  │ (query → LLM)│  │ (query → LLM)│   │
│  └──────────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI CEO AGENT (Ollama)                      │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │  Strategic Reccomendations│  │     CEO Briefing         │  │
│  │  (5 evidence-backed)     │  │  (What / Why / Next)     │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  CEO Q&A (live RAG)                       │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD                             │
│  Overview | Intelligence | Opportunities | Risks |          │
│  Sentiment | Recommendations | CEO Briefing | Q&A           │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
nvidia_agent/
├── config/
│   └── settings.py          ← ALL hyperparameters here
├── collector/
│   └── data_collector.py    ← RSS + Reddit + NewsAPI + IR
├── processor/
│   ├── knowledge_store.py   ← Clean + Chunk + Embed + ChromaDB
│   └── sentiment.py         ← HuggingFace sentiment pipeline
├── intelligence/
│   └── intel_engine.py      ← RAG → LLM → Opportunities/Risks/Trends
├── agent/
│   └── ceo_agent.py         ← Recommendations + Briefing + Q&A
├── dashboard/
│   └── app.py               ← Streamlit dashboard (7 sections)
├── data/
│   ├── chroma_db/           ← ChromaDB vector store (auto-created)
│   └── report.json          ← Latest pipeline output
├── pipeline.py              ← Master orchestrator
└── requirements.txt
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1:8b       # or qwen3:8b, mistral:7b
ollama serve                   # starts the local server
```

### 3. Run the pipeline (collects data + builds knowledge base)
```bash
python pipeline.py
# Force re-collection:
python pipeline.py --force
```

### 4. Start the dashboard
```bash
streamlit run dashboard/app.py
```

## Hyperparameter Reference

All tuneable parameters are in `config/settings.py`:

| Parameter | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | Swap to qwen3, mistral, phi-4 |
| `LLM_TEMPERATURE` | `0.3` | ↑ = more creative, ↓ = more focused |
| `LLM_MAX_TOKENS` | `1024` | Max words in LLM responses |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Swap for better quality |
| `CHUNK_SIZE` | `400` | Characters per knowledge chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `8` | Chunks retrieved per query |
| `NUM_OPPORTUNITIES` | `5` | How many to surface |
| `NUM_RISKS` | `5` | How many to surface |
| `REDDIT_POST_LIMIT` | `30` | Posts per subreddit |
| `MAX_DOCUMENTS` | `200` | Total docs cap |

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (llama3.1:8b, local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (cosine similarity, persistent) |
| Sentiment | HuggingFace Transformers (RoBERTa) |
| Dashboard | Streamlit + Plotly |
| Data | RSS (feedparser), Reddit JSON API, NewsAPI |

## AI Pipeline (RAG Flow)

```
User query / analysis task
      │
      ▼
 Embed query (SentenceTransformer)
      │
      ▼
 ChromaDB similarity search → top-k chunks
      │
      ▼
 Build context string (chunk + source + date)
      │
      ▼
 Ollama LLM (system prompt + context + task)
      │
      ▼
 Parse JSON response → structured output
      │
      ▼
 Display in dashboard
```
