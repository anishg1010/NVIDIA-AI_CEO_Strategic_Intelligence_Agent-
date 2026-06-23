"""
============================================================
  dashboard/app.py  —  NVIDIA CEO Strategic Intelligence
  REDESIGNED UI — Professional dark command-centre theme
============================================================
  Run from project root:
    streamlit run dashboard/app.py
"""

import json, os, sys
from collections import defaultdict
from datetime import datetime

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

COMPANY_NAME  = "NVIDIA"
REPORT_PATH   = os.path.join(_ROOT, "data", "report.json")

# ── must be first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="NVIDIA · CEO Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ---------- reset & base ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0d0d0f !important;
    border-right: 1px solid #1f1f23 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid #1f1f23 !important;
    color: #6b6b78 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 8px 12px !important;
    margin-bottom: 2px;
    transition: all .15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #76b900 !important;
    color: #76b900 !important;
    background: rgba(118,185,0,.06) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(118,185,0,.10) !important;
    border-color: rgba(118,185,0,.35) !important;
    color: #76b900 !important;
}

/* ---------- page background ---------- */
[data-testid="stAppViewContainer"] {
    background: #0a0a0c;
}

/* ---------- topbar card ---------- */
.topbar {
    background: #111114;
    border: 1px solid #1f1f23;
    border-radius: 12px;
    padding: 18px 24px;
    margin: 18px 0 20px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar-left { display: flex; flex-direction: column; gap: 3px; }
.topbar-title { font-size: 20px; font-weight: 600; color: #f0f0ee; letter-spacing: -.3px; }
.topbar-sub   { font-size: 12px; color: #444; }
.topbar-right { display: flex; align-items: center; gap: 14px; }
.live-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(118,185,0,.08);
    border: 1px solid rgba(118,185,0,.22);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11px; font-weight: 600; color: #76b900;
    letter-spacing: .04em;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%; background: #76b900;
}
.topbar-meta { font-size: 11px; color: #333; }

/* ---------- section label ---------- */
.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    color: #444; margin: 24px 0 12px 0;
}

/* ---------- KPI cards ---------- */
div[data-testid="stMetric"] {
    background: #111114 !important;
    border: 1px solid #1f1f23 !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
}
div[data-testid="stMetricLabel"] p {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: .12em !important; text-transform: uppercase !important;
    color: #444 !important;
}
div[data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 600 !important;
    color: #f0f0ee !important; letter-spacing: -.5px !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ---------- cards ---------- */
.card {
    background: #111114;
    border: 1px solid #1f1f23;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.card-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
    color: #444; margin-bottom: 14px;
}

/* ---------- intel rows ---------- */
.intel-row {
    padding: 13px 0;
    border-bottom: 1px solid #181820;
}
.intel-row:first-of-type { padding-top: 0; }
.intel-row:last-of-type  { border-bottom: none; padding-bottom: 0; }
.intel-head {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 12px; margin-bottom: 5px;
}
.intel-title  { font-size: 13px; font-weight: 500; color: #ddd; line-height: 1.4; }
.intel-desc   { font-size: 12px; color: #555; line-height: 1.65; margin-bottom: 9px; }
.conf-bar-wrap { display: flex; align-items: center; gap: 8px; }
.conf-track {
    flex: 1; height: 3px;
    background: #1f1f23; border-radius: 2px; overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 2px; }
.conf-pct  { font-size: 10px; color: #444; min-width: 30px; text-align: right; }
.cat-label { font-size: 10px; color: #333; margin-top: 3px; }

/* ---------- badges ---------- */
.badge {
    font-size: 9px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 5px;
    white-space: nowrap; flex-shrink: 0;
}
.b-green  { background: rgba(118,185,0,.12); color: #76b900; border: 1px solid rgba(118,185,0,.2); }
.b-red    { background: rgba(231,76,60,.12); color: #e74c3c; border: 1px solid rgba(231,76,60,.2); }
.b-orange { background: rgba(230,126,34,.12);color: #e67e22; border: 1px solid rgba(230,126,34,.2); }
.b-yellow { background: rgba(241,196,15,.12);color: #c9a600; border: 1px solid rgba(241,196,15,.2); }
.b-blue   { background: rgba(52,152,219,.12); color: #3498db; border: 1px solid rgba(52,152,219,.2); }
.b-purple { background: rgba(155,89,182,.12); color: #9b59b6; border: 1px solid rgba(155,89,182,.2); }

/* ---------- news feed ---------- */
.news-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid #181820;
}
.news-row:last-child { border-bottom: none; }
.news-dot {
    width: 7px; height: 7px; border-radius: 50%;
    margin-top: 5px; flex-shrink: 0;
}
.news-title { font-size: 12px; font-weight: 500; color: #ccc; line-height: 1.45; }
.news-meta  { font-size: 10px; color: #383838; margin-top: 3px; }

/* ---------- evidence ---------- */
.ev-block {
    border-left: 2px solid #1f1f23;
    padding: 7px 12px; margin: 5px 0;
    font-size: 12px; color: #555; line-height: 1.6;
    border-radius: 0 5px 5px 0;
}
.ev-src { font-size: 10px; color: #383838; margin-top: 3px; }

/* ---------- sentiment bars ---------- */
.sent-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.sent-label { font-size: 11px; color: #555; width: 64px; flex-shrink: 0; }
.sent-track { flex: 1; height: 5px; background: #1f1f23; border-radius: 3px; overflow: hidden; }
.sent-fill  { height: 100%; border-radius: 3px; }
.sent-pct   { font-size: 11px; color: #555; width: 36px; text-align: right; flex-shrink: 0; }

/* ---------- rec items ---------- */
.rec-wrap   { display: flex; gap: 14px; padding: 14px 0; border-bottom: 1px solid #181820; }
.rec-wrap:first-of-type { padding-top: 0; }
.rec-wrap:last-of-type  { border-bottom: none; padding-bottom: 0; }
.rec-num {
    width: 26px; height: 26px; border-radius: 50%;
    background: #181820; display: flex; align-items: center;
    justify-content: center; font-size: 11px; font-weight: 600;
    color: #555; flex-shrink: 0; margin-top: 1px;
}
.rec-title { font-size: 13px; font-weight: 500; color: #ddd; line-height: 1.4; margin-bottom: 5px; }
.rec-rationale { font-size: 12px; color: #555; line-height: 1.65; margin-bottom: 8px; }
.rec-impact {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 8px; margin-top: 10px; padding-top: 10px;
    border-top: 1px solid #181820;
}
.ri-label { font-size: 9px; color: #383838; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 2px; }
.ri-val   { font-size: 11px; color: #666; }

/* ---------- briefing ---------- */
.brief-box {
    background: #0a0a0c; border: 1px solid #1f1f23;
    border-radius: 10px; padding: 20px 24px;
    font-size: 13px; color: #777; line-height: 1.9;
    white-space: pre-wrap; font-family: 'Inter', sans-serif;
}
.brief-section {
    font-size: 10px; font-weight: 700;
    letter-spacing: .14em; color: #76b900;
}

/* ---------- Q&A ---------- */
.qa-box {
    background: #0a0a0c; border: 1px solid #1f1f23;
    border-radius: 10px; padding: 18px 22px;
    font-size: 13px; color: #777; line-height: 1.9;
    margin-top: 12px;
}
.qa-tag {
    font-size: 9px; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    color: #76b900; margin-bottom: 8px;
}

/* ---------- horizon labels ---------- */
.hz-short  { font-size: 9px; font-weight: 700; letter-spacing:.1em; color: #3498db; }
.hz-medium { font-size: 9px; font-weight: 700; letter-spacing:.1em; color: #9b59b6; }
.hz-long   { font-size: 9px; font-weight: 700; letter-spacing:.1em; color: #76b900; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1f1f23 !important;
    gap: 0 !important; margin-bottom: 16px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #444 !important;
    font-size: 12px !important; font-weight: 600 !important;
    padding: 8px 22px !important; border-radius: 0 !important;
    letter-spacing: .04em !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #76b900 !important;
    border-bottom: 2px solid #76b900 !important;
}

/* ---------- expanders ---------- */
div[data-testid="stExpander"] {
    background: #111114 !important;
    border: 1px solid #1f1f23 !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
details summary { color: #ccc !important; font-size: 13px !important; }

/* ---------- inputs ---------- */
div[data-testid="stTextInput"] input {
    background: #111114 !important;
    border-color: #1f1f23 !important;
    color: #e0e0de !important; border-radius: 8px !important;
    font-size: 13px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #111114 !important;
    border-color: #1f1f23 !important; border-radius: 8px !important;
}

/* ---------- buttons (main area) ---------- */
div[data-testid="stButton"] button {
    background: #111114 !important;
    border: 1px solid #1f1f23 !important;
    color: #888 !important; border-radius: 8px !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: .04em !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #76b900 !important; color: #76b900 !important;
    background: rgba(118,185,0,.06) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: rgba(118,185,0,.10) !important;
    border-color: rgba(118,185,0,.35) !important;
    color: #76b900 !important;
}

/* ---------- alerts ---------- */
div[data-testid="stInfo"]    { background: rgba(118,185,0,.05) !important; border-color: rgba(118,185,0,.18) !important; color: #76b900 !important; border-radius: 8px !important; }
div[data-testid="stSuccess"] { background: rgba(118,185,0,.05) !important; border-color: rgba(118,185,0,.18) !important; color: #76b900 !important; border-radius: 8px !important; }
div[data-testid="stError"]   { background: rgba(231,76,60,.05) !important; border-color: rgba(231,76,60,.18) !important; border-radius: 8px !important; }
div[data-testid="stWarning"] { background: rgba(230,126,34,.05)!important; border-color: rgba(230,126,34,.18)!important; border-radius: 8px !important; }

/* ---------- dataframe ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #1f1f23 !important; border-radius: 10px !important;
}

/* ---------- divider ---------- */
hr { border-color: #1f1f23 !important; margin: 20px 0 !important; }

/* ---------- textarea ---------- */
textarea {
    background: #0a0a0c !important; border-color: #1f1f23 !important;
    color: #777 !important; font-size: 13px !important;
    border-radius: 8px !important; line-height: 1.8 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════
def _dark_layout(h=None, t=36, b=16, l=10, r=10):
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#555", size=11, family="Inter"),
        margin=dict(t=t, b=b, l=l, r=r),
    )
    if h:
        d["height"] = h
    return d


def _find_link(ev: str, lookup: dict):
    ev_low = ev.lower()
    for tk, info in lookup.items():
        words = [w for w in tk.split() if len(w) > 4]
        if words and sum(1 for w in words if w in ev_low) >= min(2, len(words)):
            return info["url"], info["source"]
    return "", ""


def _ev_html(ev_list, lookup):
    out = ""
    for e in ev_list:
        url, src = _find_link(e, lookup)
        snippet = ". ".join(e.split(". ")[:2])
        if len(e.split(". ")) > 2:
            snippet += "."
        link = (f' <a href="{url}" target="_blank" '
                f'style="color:#333;font-size:10px;text-decoration:none">↗ {src}</a>') if url else ""
        out += f'<div class="ev-block">{snippet}{link}</div>'
    return out


def answer_ceo_question(question: str) -> str:
    try:
        import ollama
        from processor.knowledge_store import query as kb_query
        try:
            from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P
        except Exception:
            LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P = "llama3.1:8b", 0.1, 0.9

        chunks = kb_query(question, top_k=8)
        if not chunks:
            return "Knowledge base is empty. Run python pipeline.py first."

        context = "\n\n".join([f"[{c['source']} | {c['date']}]\n{c['text']}" for c in chunks])
        extra = ""
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH) as f:
                r = json.load(f)
            recs = r.get("recommendations", [])[:3]
            if recs:
                extra = "\n\nStrategic recommendations:\n"
                extra += "\n".join([f"- {rec.get('recommendation','')}" for rec in recs])

        system = (
            f"You are a strategic advisor to the CEO of {COMPANY_NAME}. "
            "Answer with evidence-based reasoning. Be concise (150-200 words). "
            "End with: Sources: [list source names]"
        )
        resp = ollama.chat(
            model=LLM_MODEL,
            options={"temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P, "num_predict": 500},
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": f"CONTEXT:\n{context}{extra}\n\nQUESTION: {question}"}],
        )
        return resp["message"]["content"]
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "refused" in msg:
            return "Ollama is not running. Fix: ollama serve"
        if "model" in msg.lower() and "not found" in msg.lower():
            return "Model not found. Fix: ollama pull llama3.1:8b"
        return f"Error: {msg}"


# ════════════════════════════════════════════════════════
#  LOAD REPORT
# ════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_report():
    if not os.path.exists(REPORT_PATH):
        return {}
    with open(REPORT_PATH) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 4px 16px;border-bottom:1px solid #1f1f23;margin-bottom:16px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;color:#76b900;margin-bottom:4px">NVDA · AI</div>
        <div style="font-size:15px;font-weight:600;color:#f0f0ee">CEO Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:9px;font-weight:700;letter-spacing:.14em;color:#333;padding:4px 0 6px;text-transform:uppercase">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("overview",     "01  Overview"),
        ("feed",         "02  Market feed"),
        ("sentiment",    "03  Sentiment"),
        ("intelligence", "04  Intelligence"),
        ("recs",         "05  Recommendations"),
        ("briefing",     "06  CEO briefing"),
        ("qa",           "07  Ask the advisor"),
    ]
    if "page" not in st.session_state:
        st.session_state.page = "overview"

    for key, label in pages:
        is_active = st.session_state.page == key
        btn_label = f"▶  {label}" if is_active else f"    {label}"
        if st.button(btn_label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:9px;font-weight:700;letter-spacing:.14em;color:#333;padding:4px 0 6px;text-transform:uppercase;border-top:1px solid #1f1f23;margin-top:4px;padding-top:16px">Pipeline</div>', unsafe_allow_html=True)

    if st.button("↺  Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    if st.button("▶  Run pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running…"):
            os.system(f'python "{os.path.join(_ROOT, "pipeline.py")}"')
        st.cache_data.clear(); st.rerun()

    if st.button("⟳  Force re-collect", use_container_width=True):
        with st.spinner("Re-collecting…"):
            os.system(f'python "{os.path.join(_ROOT, "pipeline.py")}" --force')
        st.cache_data.clear(); st.rerun()


# ════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════
report = load_report()
if not report:
    st.error("No report found. Run `python pipeline.py` then click Refresh.")
    st.stop()

meta      = report.get("meta", {})
opps      = report.get("opportunities", [])
risks     = report.get("risks", [])
trends    = report.get("trends", [])
recs      = report.get("recommendations", [])
briefing  = report.get("ceo_briefing", "")
sent      = report.get("sentiment_summary", {})
docs_meta = report.get("documents_meta", [])

url_lookup = {
    d.get("title","").lower()[:70]: {"url": d.get("url",""), "source": d.get("source","")}
    for d in docs_meta if d.get("url")
}

PAGE_TITLES = {
    "overview":     ("Overview",           "Company snapshot · live intelligence"),
    "feed":         ("Market feed",        "Live articles from 8 data sources"),
    "sentiment":    ("Sentiment analysis", "FinBERT (financial) + RoBERTa (social)"),
    "intelligence": ("Intelligence engine","RAG-powered opportunity & risk detection"),
    "recs":         ("Recommendations",    "CEO-level strategic actions · sorted by priority"),
    "briefing":     ("CEO morning briefing","What happened · Why it matters · What to do next"),
    "qa":           ("Ask the advisor",    "RAG Q&A grounded in ChromaDB · Llama 3.1:8b"),
}

page = st.session_state.page
ptitle, psub = PAGE_TITLES.get(page, ("Dashboard",""))
gen_at = meta.get("generated_at","N/A")[:16]
total_docs   = meta.get("total_documents", 0)
total_chunks = meta.get("total_chunks", 0)
num_sources  = meta.get("num_sources", 0)

# ── topbar ───────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-title">{ptitle}</div>
    <div class="topbar-sub">{psub}</div>
  </div>
  <div class="topbar-right">
    <div class="live-pill"><div class="live-dot"></div>Live · {total_docs} docs · {num_sources} sources</div>
    <div class="topbar-meta">Updated {gen_at}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "overview":
    ws    = sent.get("weighted_score", 0)
    wslbl = "Bullish" if ws > 0.05 else ("Bearish" if ws < -0.05 else "Neutral")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles collected", f"{total_docs:,}",   help="Raw documents from all 8 sources")
    c2.metric("ChromaDB chunks",    f"{total_chunks:,}",  help="Articles split into 400-char chunks for vector retrieval")
    c3.metric("Active sources",     f"{num_sources}",     help="Unique data sources contributing today")
    c4.metric("Weighted sentiment", f"{ws:+.3f}",         help=f"{wslbl} — range −1 (very bearish) to +1 (very bullish)")

    st.markdown('<div class="sec-label">Intelligence snapshot</div>', unsafe_allow_html=True)

    col_opp, col_risk = st.columns(2)

    # top opportunity
    with col_opp:
        if opps:
            o    = opps[0]
            conf = o.get("confidence_score", 0.5)
            imp  = o.get("impact_level","Medium")
            bc   = "b-green"
            st.markdown(f"""
            <div class="card">
              <div class="card-label">Top opportunity</div>
              <div class="intel-row" style="padding-top:0">
                <div class="intel-head">
                  <div class="intel-title">{o.get('title','')}</div>
                  <span class="badge {bc}">{imp}</span>
                </div>
                <div class="intel-desc">{o.get('description','')}</div>
                <div class="conf-bar-wrap">
                  <div class="conf-track"><div class="conf-fill" style="width:{int(conf*100)}%;background:#76b900"></div></div>
                  <div class="conf-pct">{conf:.0%}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # top risk
    with col_risk:
        if risks:
            r    = risks[0]
            conf = r.get("confidence_score", 0.5)
            sev  = r.get("severity","High")
            sc   = {"Critical":"#e74c3c","High":"#e67e22","Medium":"#c9a600","Low":"#76b900"}.get(sev,"#888")
            bc   = {"Critical":"b-red","High":"b-orange","Medium":"b-yellow","Low":"b-green"}.get(sev,"b-orange")
            st.markdown(f"""
            <div class="card">
              <div class="card-label">Top risk</div>
              <div class="intel-row" style="padding-top:0">
                <div class="intel-head">
                  <div class="intel-title">{r.get('title','')}</div>
                  <span class="badge {bc}">{sev}</span>
                </div>
                <div class="intel-desc">{r.get('description','')}</div>
                <div class="conf-bar-wrap">
                  <div class="conf-track"><div class="conf-fill" style="width:{int(conf*100)}%;background:{sc}"></div></div>
                  <div class="conf-pct">{conf:.0%}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # recent feed
    if docs_meta:
        rows_html = ""
        for doc in docs_meta[:10]:
            s   = doc.get("sentiment","neutral")
            dot = {"positive":"#76b900","negative":"#e74c3c","neutral":"#2a2a2a"}.get(s,"#2a2a2a")
            url = doc.get("url",""); ttl = doc.get("title","")[:95]
            src = doc.get("source",""); dt = doc.get("date","")[:10]
            lo = f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit">' if url else ""
            lc = "</a>" if url else ""
            rows_html += f"""
            <div class="news-row">
              <div class="news-dot" style="background:{dot}"></div>
              <div>
                <div class="news-title">{lo}{ttl}{lc}</div>
                <div class="news-meta">{src} · {dt}</div>
              </div>
            </div>"""
        st.markdown(f'<div class="card"><div class="card-label">Recent articles</div>{rows_html}</div>',
                    unsafe_allow_html=True)

    # source list
    if meta.get("sources"):
        with st.expander("Active data sources"):
            sc = st.columns(3)
            for i, s in enumerate(sorted(meta["sources"])):
                sc[i % 3].markdown(f'<span style="font-size:12px;color:#555">✓ {s}</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — MARKET FEED
# ════════════════════════════════════════════════════════
elif page == "feed":
    if not docs_meta:
        st.info("Run the pipeline to populate the market feed.")
    else:
        cf1, cf2, cf3 = st.columns([3, 1, 1])
        with cf1:
            search = st.text_input("", placeholder="Semantic search — e.g. AMD competition, export controls, earnings…",
                                   label_visibility="collapsed")
        with cf2:
            sf = st.selectbox("", ["All sentiments","positive","neutral","negative"],
                              label_visibility="collapsed")
        with cf3:
            all_src = sorted({d.get("source","") for d in docs_meta if d.get("source")})
            srcf    = st.selectbox("", ["All sources"] + all_src, label_visibility="collapsed")

        filtered = docs_meta
        if search and len(search.strip()) > 2:
            try:
                from processor.knowledge_store import query as kb_query
                sem  = kb_query(search.strip(), top_k=15)
                stit = {r["title"].lower()[:80] for r in sem if r.get("title")}
                surl = {r["url"] for r in sem if r.get("url")}
                matched = [d for d in docs_meta
                           if d.get("title","").lower()[:80] in stit or d.get("url","") in surl]
                filtered = matched if matched else docs_meta
                if matched:
                    st.success(f"{len(matched)} articles matched for \"{search}\"")
            except Exception as ex:
                filtered = [d for d in docs_meta if search.lower() in d.get("title","").lower()]
                st.caption(f"Semantic search unavailable — keyword fallback ({ex})")

        if sf != "All sentiments":
            filtered = [d for d in filtered if d.get("sentiment") == sf]
        if srcf != "All sources":
            filtered = [d for d in filtered if d.get("source") == srcf]

        rows_html = ""
        for doc in filtered[:30]:
            s   = doc.get("sentiment","neutral")
            dot = {"positive":"#76b900","negative":"#e74c3c","neutral":"#2a2a2a"}.get(s,"#2a2a2a")
            sc  = doc.get("sentiment_score",0)
            url = doc.get("url",""); ttl = doc.get("title","")[:100]
            src = doc.get("source",""); dt = doc.get("date","")[:10]
            lo  = f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit">' if url else ""
            lc  = "</a>" if url else ""
            rows_html += f"""
            <div class="news-row">
              <div class="news-dot" style="background:{dot}"></div>
              <div>
                <div class="news-title">{lo}{ttl}{lc}</div>
                <div class="news-meta">{src} · {dt} · score {sc:.2f}</div>
              </div>
            </div>"""
        st.markdown(
            f'<div class="card"><div class="card-label">{min(30,len(filtered))} of {len(filtered)} articles</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — SENTIMENT
# ════════════════════════════════════════════════════════
elif page == "sentiment":
    if not sent:
        st.info("No sentiment data. Run the pipeline first.")
    else:
        ws     = sent.get("weighted_score",0)
        wslbl  = "Bullish" if ws>.1 else ("Bearish" if ws<-.1 else "Neutral")
        ns     = sent.get("news_sentiment",{})
        ps     = sent.get("public_sentiment",{})
        n_avg  = ns.get("avg_score",0.0)
        p_avg  = ps.get("avg_score",0.0)

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Overall mood",   wslbl)
        k2.metric("Positive",       f"{sent.get('positive_pct',0)}%", sent.get("positive",0))
        k3.metric("Neutral",        f"{sent.get('neutral_pct',0)}%",  sent.get("neutral",0))
        k4.metric("Negative",       f"{sent.get('negative_pct',0)}%", sent.get("negative",0))
        k5.metric("Weighted score", f"{ws:+.3f}", wslbl)

        st.markdown('<div class="sec-label">Breakdown</div>', unsafe_allow_html=True)

        cl, cr = st.columns(2)

        # donut
        with cl:
            fd = go.Figure(go.Pie(
                labels=["Positive","Neutral","Negative"],
                values=[sent.get("positive",0), sent.get("neutral",0), sent.get("negative",0)],
                hole=.6, marker_colors=["#76b900","#1f1f23","#e74c3c"],
                textinfo="label+percent",
                textfont=dict(color="#777",size=11),
            ))
            fd.update_layout(title=dict(text="Sentiment split",font=dict(color="#444",size=12)),
                             showlegend=False, **_dark_layout(280))
            st.plotly_chart(fd, use_container_width=True)

        # gauge
        with cr:
            bc = "#76b900" if ws>0 else "#e74c3c"
            fg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ws,
                number={"valueformat":"+.3f","font":{"color":bc,"size":28,"family":"Inter"}},
                gauge={
                    "axis":{"range":[-1,1],"tickwidth":1,"tickcolor":"#2a2a2a",
                            "tickfont":{"color":"#444","size":10}},
                    "bar":{"color":bc,"thickness":.22},
                    "bgcolor":"rgba(0,0,0,0)","bordercolor":"#1f1f23",
                    "steps":[
                        {"range":[-1,-.3],"color":"rgba(231,76,60,.07)"},
                        {"range":[-.3,.3],"color":"rgba(255,255,255,.02)"},
                        {"range":[.3,1], "color":"rgba(118,185,0,.07)"},
                    ],
                    "threshold":{"line":{"color":"#333","width":2},"thickness":.75,"value":0},
                },
                title={"text":f"Weighted score · {wslbl}","font":{"color":"#444","size":12}},
            ))
            fg.update_layout(**_dark_layout(280, t=60, l=30, r=30))
            st.plotly_chart(fg, use_container_width=True)

        # stacked bar by source
        if docs_meta:
            ss = defaultdict(lambda:{"positive":0,"neutral":0,"negative":0})
            for d in docs_meta:
                ss[d.get("source","?")[:22]][d.get("sentiment","neutral")] += 1
            si  = sorted(ss.items(), key=lambda x:sum(x[1].values()), reverse=True)[:10]
            src = [s for s,_ in si]
            fb  = go.Figure()
            fb.add_trace(go.Bar(name="Positive",x=src,y=[v["positive"] for _,v in si],marker_color="#76b900"))
            fb.add_trace(go.Bar(name="Neutral", x=src,y=[v["neutral"]  for _,v in si],marker_color="#1f1f23"))
            fb.add_trace(go.Bar(name="Negative",x=src,y=[v["negative"] for _,v in si],marker_color="#e74c3c"))
            fb.update_layout(
                title=dict(text="Sentiment by source",font=dict(color="#444",size=12)),
                barmode="stack",
                legend=dict(orientation="h",y=-0.3,font=dict(color="#555",size=10)),
                xaxis=dict(tickangle=-35,tickfont=dict(color="#444",size=10),gridcolor="#181820"),
                yaxis=dict(tickfont=dict(color="#444",size=10),gridcolor="#181820"),
                **_dark_layout(300)
            )
            st.plotly_chart(fb, use_container_width=True)

        # news vs public + model usage
        cl2, cr2 = st.columns(2)
        with cl2:
            if ps.get("count",0) > 0:
                fc = go.Figure(go.Bar(
                    x=["News (FinBERT)","Public (RoBERTa)"],
                    y=[n_avg, p_avg],
                    marker_color=["#76b900" if n_avg>=0 else "#e74c3c",
                                  "#3498db" if p_avg>=0 else "#e74c3c"],
                    text=[f"{n_avg:+.3f}",f"{p_avg:+.3f}"],
                    textposition="outside",
                    textfont=dict(color="#777",size=11),
                ))
                fc.add_hline(y=0,line_dash="dot",line_color="#2a2a2a",line_width=1)
                fc.update_layout(
                    title=dict(text="News vs public score",font=dict(color="#444",size=12)),
                    xaxis=dict(tickfont=dict(color="#555",size=11),gridcolor="#181820"),
                    yaxis=dict(range=[-1.2,1.2],tickfont=dict(color="#444",size=10),gridcolor="#181820"),
                    **_dark_layout(260)
                )
                st.plotly_chart(fc, use_container_width=True)

        with cr2:
            mb   = sent.get("model_breakdown",{})
            fb_n = mb.get("FinBERT",0); rb_n = mb.get("RoBERTa",0)
            tot  = fb_n + rb_n or 1
            st.markdown(f"""
            <div class="card" style="margin-top:0">
              <div class="card-label">Model usage</div>
              <div class="sent-row">
                <div class="sent-label">FinBERT</div>
                <div class="sent-track"><div class="sent-fill" style="width:{fb_n/tot*100:.0f}%;background:#185FA5"></div></div>
                <div class="sent-pct">{fb_n}</div>
              </div>
              <div class="sent-row">
                <div class="sent-label">RoBERTa</div>
                <div class="sent-track"><div class="sent-fill" style="width:{rb_n/tot*100:.0f}%;background:#9b59b6"></div></div>
                <div class="sent-pct">{rb_n}</div>
              </div>
              <div style="font-size:11px;color:#333;margin-top:12px;line-height:1.7">
                FinBERT → financial sources (Yahoo Finance, NVIDIA IR)<br>
                RoBERTa → social sources (Reddit, HackerNews)
              </div>
            </div>
            """, unsafe_allow_html=True)

        # trend chart
        st.markdown('<div class="sec-label">30-day sentiment trend</div>', unsafe_allow_html=True)
        ts = sent.get("trend_series",[])
        if ts and len(ts) >= 2:
            dates  = [p["date"]  for p in ts]
            scores = [p["score"] for p in ts]
            labels = [p["label"] for p in ts]
            mcolors = ["#76b900" if l=="positive" else ("#e74c3c" if l=="negative" else "#333") for l in labels]

            ft = go.Figure()
            ft.add_hrect(y0=-.05, y1=.05, fillcolor="rgba(255,255,255,.015)", line_width=0)
            ft.add_trace(go.Scatter(
                x=dates, y=scores, mode="lines+markers",
                line=dict(color="#76b900",width=2),
                marker=dict(size=7,color=mcolors,line=dict(width=1.5,color="#0a0a0c")),
                fill="tozeroy", fillcolor="rgba(118,185,0,.05)",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))
            if len(scores) >= 4:
                mom = (sum(scores[-2:])/2) - (sum(scores[:2])/2)
                mt  = "↑ improving" if mom>.02 else ("↓ declining" if mom<-.02 else "→ stable")
                ft.add_annotation(x=dates[-1],y=scores[-1],text=mt,showarrow=True,arrowhead=2,
                                  font=dict(size=11,color="#76b900"),arrowcolor="#76b900")
            ft.update_layout(
                xaxis=dict(tickangle=-30,tickfont=dict(color="#444",size=10),gridcolor="#181820"),
                yaxis=dict(range=[-1.1,1.1],tickfont=dict(color="#444",size=10),gridcolor="#181820"),
                hovermode="x unified", **_dark_layout(300)
            )
            st.plotly_chart(ft, use_container_width=True)

            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Data points", len(scores))
            t2.metric("Mean score",  f"{float(np.mean(scores)):+.3f}")
            t3.metric("Volatility",  f"{float(np.std(scores)):.3f}", help="Std dev — higher = more swings")
            t4.metric("Latest",      f"{scores[-1]:+.3f}",
                      f"{'↑' if scores[-1]>scores[-2] else '↓'} vs prev" if len(scores)>=2 else "")
        else:
            st.info("Trend chart needs ≥2 dates of data. Re-run the pipeline on separate days.")

        # top articles
        st.markdown('<div class="sec-label">Most positive · Most negative</div>', unsafe_allow_html=True)
        cp, cn = st.columns(2)
        with cp:
            for a in sent.get("top_positive",[])[:5]:
                url = a.get("url",""); ttl = a.get("title","")[:70]
                lo = f'[{ttl}…]({url})' if url else f'{ttl}…'
                st.markdown(f'<div style="font-size:12px;color:#ccc;margin-bottom:2px">{"["+ttl+"…]("+url+")" if url else ttl+"…"}</div><div style="font-size:10px;color:#383838;margin-bottom:8px">{a.get("source","")} — {a.get("score",0):.2f}</div>', unsafe_allow_html=True)
        with cn:
            for a in sent.get("top_negative",[])[:5]:
                url = a.get("url",""); ttl = a.get("title","")[:70]
                st.markdown(f'<div style="font-size:12px;color:#ccc;margin-bottom:2px">{"["+ttl+"…]("+url+")" if url else ttl+"…"}</div><div style="font-size:10px;color:#383838;margin-bottom:8px">{a.get("source","")} — {a.get("score",0):.2f}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — INTELLIGENCE ENGINE
# ════════════════════════════════════════════════════════
elif page == "intelligence":
    tab_o, tab_r, tab_t = st.tabs(["  Opportunities  ", "  Risks  ", "  Trends  "])

    with tab_o:
        if opps:
            rows_html = ""
            for o in opps:
                conf = o.get("confidence_score",.5)
                imp  = o.get("impact_level","Medium")
                ev   = _ev_html(o.get("evidence",[]), url_lookup)
                rows_html += f"""
                <div class="intel-row">
                  <div class="intel-head">
                    <div class="intel-title">{o.get('title','')}</div>
                    <span class="badge b-green">{imp}</span>
                  </div>
                  <div class="intel-desc">{o.get('description','')}</div>
                  {ev}
                  <div class="conf-bar-wrap" style="margin-top:8px">
                    <div class="conf-track"><div class="conf-fill" style="width:{int(conf*100)}%;background:#76b900"></div></div>
                    <div class="conf-pct">{conf:.0%} · {o.get('category','')}</div>
                  </div>
                </div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No opportunities detected. Run the pipeline.")

    with tab_r:
        if risks:
            sc_map  = {"Critical":"#e74c3c","High":"#e67e22","Medium":"#c9a600","Low":"#76b900"}
            bc_map  = {"Critical":"b-red","High":"b-orange","Medium":"b-yellow","Low":"b-green"}
            rows_html = ""
            for r in risks:
                conf = r.get("confidence_score",.5)
                sev  = r.get("severity","High")
                sc   = sc_map.get(sev,"#888"); bc = bc_map.get(sev,"b-orange")
                ev   = _ev_html(r.get("evidence",[]), url_lookup)
                rows_html += f"""
                <div class="intel-row">
                  <div class="intel-head">
                    <div class="intel-title">{r.get('title','')}</div>
                    <span class="badge {bc}">{sev}</span>
                  </div>
                  <div class="intel-desc">{r.get('description','')}</div>
                  {ev}
                  <div class="conf-bar-wrap" style="margin-top:8px">
                    <div class="conf-track"><div class="conf-fill" style="width:{int(conf*100)}%;background:{sc}"></div></div>
                    <div class="conf-pct">{conf:.0%} · {r.get('category','')}</div>
                  </div>
                </div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No risks detected. Run the pipeline.")

    with tab_t:
        if trends:
            rows_html = ""
            for t in trends:
                h = t.get("time_horizon","")
                hc = "hz-short" if "Short" in h else ("hz-long" if "Long" in h else "hz-medium")
                hl = "SHORT-TERM" if "Short" in h else ("LONG-TERM" if "Long" in h else "MEDIUM-TERM")
                ev = _ev_html(t.get("evidence",[]), url_lookup)
                rows_html += f"""
                <div class="intel-row">
                  <div class="intel-head">
                    <div>
                      <div class="{hc}" style="margin-bottom:3px">{hl}</div>
                      <div class="intel-title">{t.get('title','')}</div>
                    </div>
                    <span class="badge b-blue">{t.get('relevance','Medium')}</span>
                  </div>
                  <div class="intel-desc">{t.get('description','')}</div>
                  {ev}
                  <div class="cat-label">{t.get('category','')}</div>
                </div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No trends detected. Run the pipeline.")


# ════════════════════════════════════════════════════════
#  PAGE — RECOMMENDATIONS
# ════════════════════════════════════════════════════════
elif page == "recs":
    if not recs:
        st.info("No recommendations yet. Run the pipeline.")
    else:
        hc = sum(1 for r in recs if r.get("priority")=="High")
        mc = sum(1 for r in recs if r.get("priority")=="Medium")
        lc = sum(1 for r in recs if r.get("priority")=="Low")
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:16px">
          <span class="badge b-red">{hc} High</span>
          <span class="badge b-orange">{mc} Medium</span>
          <span class="badge b-green">{lc} Low</span>
        </div>""", unsafe_allow_html=True)

        rows_html = ""
        for i, rec in enumerate(recs, 1):
            pri   = rec.get("priority","Medium")
            pc    = {"High":"b-red","Medium":"b-orange","Low":"b-green"}.get(pri,"b-orange")
            rl    = rec.get("risk_level","Medium")
            rc    = {"High":"b-red","Medium":"b-orange","Low":"b-green"}.get(rl,"b-orange")
            imp   = rec.get("expected_impact",{})
            ra    = rec.get("risk_assessment",{})
            ev    = _ev_html(rec.get("supporting_evidence",[]), url_lookup)
            rows_html += f"""
            <div class="rec-wrap">
              <div class="rec-num">{i}</div>
              <div style="flex:1">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:6px">
                  <div class="rec-title">{rec.get('recommendation','')}</div>
                  <span class="badge {pc}" style="flex-shrink:0">{pri}</span>
                </div>
                <div class="rec-rationale">{rec.get('rationale','')}</div>
                {ev}
                <div class="rec-impact">
                  <div><div class="ri-label">Revenue impact</div><div class="ri-val">{imp.get('revenue','N/A')}</div></div>
                  <div><div class="ri-label">Timeline</div><div class="ri-val">{imp.get('timeline','N/A')}</div></div>
                  <div><div class="ri-label">Risk level</div><span class="badge {rc}">{rl}</span></div>
                </div>
              </div>
            </div>"""
        st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — CEO BRIEFING
# ════════════════════════════════════════════════════════
elif page == "briefing":
    cb, ct = st.columns([3, 1], gap="medium")

    with cb:
        if briefing:
            formatted = briefing
            for s in ["WHAT HAPPENED","WHY IT MATTERS","WHAT TO DO NEXT"]:
                formatted = formatted.replace(s, f'<span class="brief-section">{s}</span>')
            st.markdown(f"""
            <div class="card">
              <div class="card-label">Morning briefing</div>
              <div class="brief-box">{formatted}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No briefing generated. Run the pipeline.")

    with ct:
        if trends:
            rows_html = ""
            for t in trends:
                h  = t.get("time_horizon","")
                hc = "hz-short" if "Short" in h else ("hz-long" if "Long" in h else "hz-medium")
                hl = "SHORT-TERM" if "Short" in h else ("LONG-TERM" if "Long" in h else "MEDIUM-TERM")
                rows_html += f"""
                <div class="intel-row">
                  <div class="{hc}" style="margin-bottom:3px">{hl}</div>
                  <div class="intel-title" style="font-size:12px">{t.get('title','')}</div>
                  <div class="intel-desc" style="font-size:11px;margin-top:3px">{t.get('description','')[:100]}…</div>
                </div>"""
            st.markdown(f'<div class="card"><div class="card-label">Emerging trends</div>{rows_html}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE — CEO Q&A
# ════════════════════════════════════════════════════════
elif page == "qa":
    suggested = [
        "What are AMD and Intel doing in AI chips?",
        "What is NVIDIA's biggest risk right now?",
        "Should NVIDIA expand into automotive AI?",
        "What does sentiment say about investor confidence?",
        "What should NVIDIA prioritize this quarter?",
    ]
    sq_cols = st.columns(3)
    for i, sq in enumerate(suggested):
        if sq_cols[i % 3].button(sq, key=f"sq_{i}"):
            st.session_state["ceo_q"] = sq

    question = st.text_input(
        "", value=st.session_state.get("ceo_q",""),
        placeholder="Ask a strategic question — e.g. What is NVIDIA's biggest risk right now?",
        label_visibility="collapsed",
    )

    if st.button("Get AI answer", type="primary") and question:
        with st.spinner("Searching knowledge base · reasoning with Llama 3.1…"):
            answer = answer_ceo_question(question)
        st.markdown(f"""
        <div class="qa-box">
          <div class="qa-tag">AI advisor · grounded in ChromaDB</div>
          {answer}
        </div>""", unsafe_allow_html=True)
        if "ceo_q" in st.session_state:
            del st.session_state["ceo_q"]

    st.markdown('<div class="sec-label" style="margin-top:24px">How this works</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    for col, num, head, desc in [
        (h1, "01", "Query ChromaDB",
         "Top 8 chunks retrieved by cosine similarity across all stored vectors"),
        (h2, "02", "Build context",
         "Chunks + top 3 recommendations injected into the Llama 3.1:8b prompt"),
        (h3, "03", "Generate answer",
         "150–200 word grounded response at temperature 0.1 with source attribution"),
    ]:
        col.markdown(f"""
        <div class="card" style="margin-top:0">
          <div style="font-size:28px;font-weight:700;color:#1f1f23;margin-bottom:8px">{num}</div>
          <div style="font-size:12px;font-weight:600;color:#666;margin-bottom:5px">{head}</div>
          <div style="font-size:11px;color:#444;line-height:1.65">{desc}</div>
        </div>""", unsafe_allow_html=True)

