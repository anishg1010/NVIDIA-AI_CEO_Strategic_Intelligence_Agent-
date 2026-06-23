"""
============================================================
  config/settings.py — Master Configuration
  NVIDIA AI CEO Strategic Intelligence Agent
============================================================
  Every tuneable parameter in one place.
  Technical reasoning documented inline.
"""

import os

# ─────────────────────────────────────────────────────────
#  LLM — Ollama (local)
#
#  LLM_TEMPERATURE = 0.1  →  near-deterministic JSON output.
#    Raising above 0.3 causes malformed JSON in structured tasks.
#  LLM_MAX_TOKENS  = 2048 →  5 JSON objects × ~200 tokens each
#    = ~1000 tokens minimum. 1024 truncates the closing "]".
# ─────────────────────────────────────────────────────────
LLM_MODEL        = "llama3.1:8b"
LLM_TEMPERATURE  = 0.1
LLM_TOP_P        = 0.9
LLM_MAX_TOKENS   = 2048          # was 1024 — raised to prevent JSON truncation
OLLAMA_BASE_URL  = "http://localhost:11434"

# ─────────────────────────────────────────────────────────
#  Embeddings
#  BAAI/bge-base-en-v1.5: 768-dim, MTEB score 63.6
#  all-MiniLM-L6-v2:      384-dim, MTEB score 56.3, 2× faster
# ─────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM    = 768

# ─────────────────────────────────────────────────────────
#  ChromaDB
#  TOP_K_RETRIEVAL = 8  →  8 chunks × 400 chars ≈ 800 tokens
#    of context. Safe within llama3.1:8b's 8192-token window.
# ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./data/chroma_db"
CHROMA_COLLECTION  = "nvidia_intelligence"
TOP_K_RETRIEVAL    = 8                    # FIX: was blank — syntax error

# ─────────────────────────────────────────────────────────
#  Sentiment Models
#  FinBERT  → financial domain (earnings, IR, analyst reports)
#  RoBERTa  → social/general domain (Reddit, HN, StockTwits)
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
#  Sentiment Models
#
#  Both models are FREE, PUBLIC, and require NO HuggingFace
#  token or login. They have been stable since 2020-2021.
#
#  FINBERT_MODEL = "ProsusAI/finbert"
#    The original FinBERT. Trained on 10,000 financial news
#    sentences from Reuters. Labels: positive/neutral/negative.
#    Always public — never gated. ~440MB download, cached after.
#    DO NOT use "yiyanghkust/finbert-tone" — it is now 401'ing.
#
#  SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
#    RoBERTa trained on 124M tweets. Labels: LABEL_0/1/2
#    (mapped to negative/neutral/positive by _norm() in sentiment.py).
#    Stable public version — DO NOT use "...-latest" variant
#    which now requires an expired HF access token.
# ─────────────────────────────────────────────────────────
FINBERT_MODEL   = "ProsusAI/finbert"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# ─────────────────────────────────────────────────────────
#  Data Collection
# ─────────────────────────────────────────────────────────
COMPANY_NAME    = "NVIDIA"
COMPANY_TICKER  = "NVDA"

RSS_FEEDS = [
    "https://nvidianews.nvidia.com/rss/all",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    "https://techcrunch.com/feed/",             # real TechCrunch (not blogspot scraper)
    "https://www.theverge.com/rss/index.xml",
    "https://www.electronicdesign.com/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
    "https://www.tomshardware.com/feeds/all",
]

REDDIT_SUBREDDITS = [
    "nvidia", "artificial", "MachineLearning",
    "stocks", "hardware", "Futurology",
]
REDDIT_POST_LIMIT = 25

NEWS_API_KEY     = os.environ.get("NEWS_API_KEY", "d18749108ef94458aa73b8f16e10f6cc")
NEWS_QUERY       = "NVIDIA GPU AI semiconductor"
NEWS_MAX_RESULTS = 50

HACKERNEWS_QUERY = "NVIDIA"
HACKERNEWS_LIMIT = 30

MAX_DOCUMENTS    = 300    # FIX: was 10 — only collected Yahoo Finance
                          # 300 gives ~200 news + ~80 social + context docs

# ─────────────────────────────────────────────────────────
#  Text Chunking
#  CHUNK_SIZE = 400 → ~100 words per chunk, coherent sentences.
#    Was 100 chars (25 words) — too small for meaningful context.
#  CHUNK_OVERLAP = 80 → 20% overlap prevents split-sentence loss.
# ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 400    # FIX: was 100 — too small for coherent RAG context
CHUNK_OVERLAP = 80

# ─────────────────────────────────────────────────────────
#  Intelligence Engine
# ─────────────────────────────────────────────────────────
OPPORTUNITY_THRESHOLD = 0.65
RISK_THRESHOLD        = 0.65
NUM_OPPORTUNITIES     = 5    # FIX: was 1
NUM_RISKS             = 5    # FIX: was 1
NUM_TRENDS            = 5    # FIX: was 1

# ─────────────────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────────────────
DASHBOARD_TITLE      = "NVIDIA — CEO Strategic Intelligence Dashboard"
AUTO_REFRESH_SECONDS = 300
