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
│  │ Clean Text  │→ │ Chunk (400)  │→ │ Embed (BAAI/bge base)     │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           ChromaDB (cosine similarity)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────┐                                     │
│  │ Sentiment (RoBERTa)  │                                     │          (Finbert)
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

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                          │
│                         data_collector.py                             │
│                                                                        │
│  RSS(8) ─┐                                                             │
│  Reddit ─┤                                                             │
│  StockTwits─┤  collect_all()  ──►  relevance filter  ──►  dedup       │
│  NewsAPI ─┤  (9 collectors)       (35 NVIDIA keywords)  (title+URL)   │
│  NVIDIA IR─┤                                                           │
│  HackerNews┤                                                           │
│  GitHub ──┤                                                            │
│  Yahoo ───┤                                                            │
│  Wikipedia┘                                                            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  list[dict]  ~244 articles
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER                                │
│                  knowledge_store.py + sentiment.py                    │
│                                                                        │
│  clean_text()     chunk_text()      embed()          ChromaDB         │
│  strip HTML   ──► 400 chars    ──►  BAAI/bge     ──► cosine sim      │
│  remove URLs      80 overlap        768-dim          ~3395 chunks     │
│                                                                        │
│  sentiment.py (parallel):                                             │
│  FinBERT (financial sources) ──► positive/neutral/negative + score   │
│  Twitter-RoBERTa (social)    ──► positive/neutral/negative + score   │
│  VADER (fallback)            ──► compound score                       │
└──────────┬───────────────────────────────┬───────────────────────────┘
           │  3395 chunks stored           │  sentiment on all 244 docs
           ▼                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE ENGINE (RAG)                           │
│                         intel_engine.py                                │
│                                                                        │
│  For each category (opportunities / risks / trends):                  │
│                                                                        │
│  search query ──► embed query ──► cosine search ──► top-8 chunks     │
│       │           (768-dim)       ChromaDB            with source      │
│       │                                ▼                               │
│       └────────────────────►  build context string                    │
│                                  [Source N: name | date]              │
│                                         ▼                              │
│                              Ollama llama3.1:8b                       │
│                              temp=0.1, system+user prompt             │
│                                         ▼                              │
│                              parse JSON response                       │
│                              5 items × 3 categories = 15 insights     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  intelligence dict
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         AI CEO AGENT                                   │
│                           ceo_agent.py                                 │
│                                                                        │
│  generate_recommendations()  ──► 5 priority-sorted strategic actions  │
│  generate_ceo_briefing()     ──► 300-word executive summary           │
│  answer_ceo_question()       ──► live RAG Q&A per question            │
│                                         ▼                              │
│                              report.json  (full pipeline output)       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  report.json
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   EXECUTIVE INTELLIGENCE DASHBOARD                     │
│                           dashboard/app.py                             │
│                           Streamlit + Plotly                           │
│                         @cache_data(ttl=300)                           │
│                                                                        │
│  S1: Overview    │ S2: Market Intel  │ S3: Opportunities              │
│  244 docs        │ semantic search   │ 5 items, evidence+links        │
│  3395 chunks     │ ChromaDB query    │ confidence scores              │
│  39 sources      │ sentiment filter  │                                │
│──────────────────┼───────────────────┼────────────────────────────── │
│  S4: Risks       │ S5: Sentiment     │ S6: Recommendations            │
│  severity table  │ 5.1 News FinBERT  │ priority-sorted                │
│  category filter │ 5.2 Public RoBERTa│ expected impact                │
│  evidence links  │ 5.3 Trends chart  │ risk assessment                │
│──────────────────┼───────────────────┼────────────────────────────── │
│  S7: CEO Briefing (What/Why/Next) │ CEO Q&A (live RAG per question)  │
└──────────────────────────────────────────────────────────────────────┘
                                   ▲
┌──────────────────────────────────────────────────────────────────────┐
│                      PIPELINE ORCHESTRATOR                             │
│                           pipeline.py                                  │
│                                                                        │
│  cache check ──► collect ──► sentiment ──► store ──► intel ──► agent  │
│  --force flag    Step 1       Step 2       Step 3   Step 4   Step 5   │
└──────────────────────────────────────────────────────────────────────┘
                                   ▲
┌──────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION                                 │
│                        config/settings.py                              │
│           Single source of truth for all hyperparameters              │
│  LLM_MODEL │ CHUNK_SIZE │ TOP_K_RETRIEVAL │ FINBERT_MODEL │ ...       │
└──────────────────────────────────────────────────────────────────────┘
```

---



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

| Layer | Technology | Why |
|---|---|---|
| LLM | Ollama llama3.1:8b (local) | Open-source, no API costs, 8192-token context |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim) | MTEB score 63.6 vs 56.3 for MiniLM |
| Vector DB | ChromaDB (cosine similarity, persistent) | Local, persistent across runs, fast cosine search |
| Sentiment (financial) | ProsusAI/FinBERT | Trained on Reuters financial news |
| Sentiment (social) | Cardiff NLP Twitter-RoBERTa | Trained on 124M tweets |
| Sentiment (fallback) | VADER | Rule-based, offline, no download needed |
| Dashboard | Streamlit + Plotly | Rapid prototyping, interactive charts |
| Data sources | RSS, Reddit API, NewsAPI, HackerNews Algolia, StockTwits | Real-time, diverse, free |


## AI Pipeline (RAG Flow)

```
CEO question or intelligence task
          │
          ▼
 Embed query → 768-dim vector
   (BAAI/bge-base-en-v1.5)
          │
          ▼
 ChromaDB cosine similarity search
   top-8 chunks from ~3395 stored
          │
          ▼
 Build context string
   [Source N: name | date]\ntext
          │
          ▼
 Ollama llama3.1:8b
   system prompt + context + task
   temperature=0.1 (near-deterministic)
          │
          ▼
 Parse JSON response
   structured output: title, description,
   confidence_score, evidence[], category
          │
          ▼
 Display in dashboard with evidence links
```

```
