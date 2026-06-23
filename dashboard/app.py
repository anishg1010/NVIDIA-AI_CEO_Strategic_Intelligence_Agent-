"""
NVIDIA CEO Strategic Intelligence Dashboard
Premium redesign — run from project root:
  streamlit run dashboard/app.py
"""
import json, os, sys
from collections import defaultdict

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

COMPANY_NAME = "NVIDIA"
REPORT_PATH  = os.path.join(_ROOT, "data", "report.json")

st.set_page_config(
    page_title="NVIDIA · CEO Intelligence",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 28px 40px 28px !important;
  max-width: 100% !important;
}

/* ─── App bg ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #08080a; }

/* ─── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0c0c0f !important;
  border-right: 1px solid rgba(255,255,255,.055) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  background: transparent !important;
  border: none !important;
  color: #4a4a58 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-align: left !important;
  padding: 9px 20px !important;
  border-radius: 0 !important;
  letter-spacing: .01em;
  border-left: 2px solid transparent !important;
  transition: color .15s, border-color .15s, background .15s;
  box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  color: #c8c8d0 !important;
  background: rgba(255,255,255,.03) !important;
  border-left-color: rgba(255,255,255,.1) !important;
}

/* active page — injected via data-active attr trick */
[data-testid="stSidebar"] .stButton > button[data-active="true"] {
  color: #76b900 !important;
  background: rgba(118,185,0,.06) !important;
  border-left-color: #76b900 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: rgba(118,185,0,.08) !important;
  border: 1px solid rgba(118,185,0,.25) !important;
  border-left: 2px solid rgba(118,185,0,.4) !important;
  color: #76b900 !important;
  border-radius: 6px !important;
  margin: 2px 16px !important;
  width: calc(100% - 32px) !important;
  font-size: 11px !important;
  letter-spacing: .04em !important;
}

/* ─── Page header ────────────────────────────────────────── */
.ph {
  display: flex; align-items: flex-end;
  justify-content: space-between;
  padding: 28px 0 20px;
  border-bottom: 1px solid rgba(255,255,255,.06);
  margin-bottom: 24px;
}
.ph-left {}
.ph-eyebrow {
  font-size: 10px; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase;
  color: #76b900; margin-bottom: 5px;
}
.ph-title {
  font-size: 22px; font-weight: 600;
  color: #f2f2f0; letter-spacing: -.4px; line-height: 1.15;
}
.ph-right { display: flex; align-items: center; gap: 16px; }
.ph-stat {
  text-align: right;
}
.ph-stat-val {
  font-size: 18px; font-weight: 600;
  color: #f2f2f0; letter-spacing: -.3px;
  font-family: 'JetBrains Mono', monospace;
}
.ph-stat-lbl {
  font-size: 9px; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; color: #38383f;
}
.ph-divider {
  width: 1px; height: 32px;
  background: rgba(255,255,255,.07);
}
.status-badge {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(118,185,0,.07);
  border: 1px solid rgba(118,185,0,.2);
  border-radius: 999px; padding: 6px 14px;
  font-size: 10px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: #76b900;
}
.status-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #76b900;
  box-shadow: 0 0 6px #76b900;
}

/* ─── KPI cards ──────────────────────────────────────────── */
div[data-testid="stMetric"] {
  background: #0e0e11 !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 10px !important;
  padding: 18px 20px !important;
  position: relative;
  overflow: hidden;
}
div[data-testid="stMetric"]::after {
  content: '';
  position: absolute; top: 0; left: 0;
  width: 100%; height: 2px;
  background: linear-gradient(90deg, rgba(118,185,0,.5), transparent);
}
div[data-testid="stMetricLabel"] p {
  font-size: 9px !important; font-weight: 700 !important;
  letter-spacing: .15em !important; text-transform: uppercase !important;
  color: #38383f !important; margin-bottom: 6px !important;
}
div[data-testid="stMetricValue"] {
  font-size: 28px !important; font-weight: 600 !important;
  color: #f2f2f0 !important; letter-spacing: -.5px !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ─── Content cards ──────────────────────────────────────── */
.gcard {
  background: #0e0e11;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 12px;
  padding: 20px 22px;
  margin-bottom: 12px;
}
.gcard-title {
  font-size: 9px; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase;
  color: #38383f; margin-bottom: 16px;
}

/* ─── Intel rows ─────────────────────────────────────────── */
.irow {
  padding: 14px 0;
  border-bottom: 1px solid rgba(255,255,255,.04);
}
.irow:first-child { padding-top: 0; }
.irow:last-child  { border-bottom: none; padding-bottom: 0; }
.irow-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
.irow-name { font-size: 13px; font-weight: 500; color: #d8d8d5; line-height: 1.4; }
.irow-desc { font-size: 12px; color: #48484f; line-height: 1.7; margin-bottom: 10px; }
.cbar-wrap { display: flex; align-items: center; gap: 8px; }
.cbar-track { flex:1; height:2px; background:rgba(255,255,255,.06); border-radius:1px; overflow:hidden; }
.cbar-fill  { height:100%; border-radius:1px; }
.cbar-pct   { font-size:9px; color:#38383f; width:28px; text-align:right;
               font-family:'JetBrains Mono',monospace; }
.irow-cat   { font-size:9px; color:#2e2e35; margin-top:4px; letter-spacing:.08em; }

/* ─── Badge ──────────────────────────────────────────────── */
.gbd {
  font-size: 9px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 4px;
  white-space: nowrap; flex-shrink: 0;
  border: 1px solid;
}
.gbd-g { color:#76b900; background:rgba(118,185,0,.09); border-color:rgba(118,185,0,.2); }
.gbd-r { color:#e74c3c; background:rgba(231,76,60,.09);  border-color:rgba(231,76,60,.2); }
.gbd-o { color:#e67e22; background:rgba(230,126,34,.09); border-color:rgba(230,126,34,.2); }
.gbd-y { color:#c9a600; background:rgba(241,196,15,.09); border-color:rgba(241,196,15,.2); }
.gbd-b { color:#3b82f6; background:rgba(59,130,246,.09); border-color:rgba(59,130,246,.2); }
.gbd-p { color:#9b59b6; background:rgba(155,89,182,.09); border-color:rgba(155,89,182,.2); }

/* ─── Evidence block ─────────────────────────────────────── */
.ev {
  background: rgba(255,255,255,.02);
  border-left: 1px solid rgba(255,255,255,.08);
  padding: 8px 12px; margin: 5px 0;
  font-size: 11px; color: #48484f;
  line-height: 1.65; border-radius: 0 5px 5px 0;
}
.ev a { color: #2e2e35; font-size: 10px; text-decoration: none; }

/* ─── News feed ──────────────────────────────────────────── */
.nrow { display:flex; align-items:flex-start; gap:12px; padding:11px 0;
        border-bottom:1px solid rgba(255,255,255,.04); }
.nrow:last-child { border-bottom:none; }
.ndot { width:6px; height:6px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.ntitle { font-size:12px; font-weight:500; color:#c0c0bd; line-height:1.45; }
.nmeta  { font-size:10px; color:#2e2e35; margin-top:3px; letter-spacing:.01em; }

/* ─── Sentiment bars ─────────────────────────────────────── */
.srow { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.slbl { font-size:10px; color:#48484f; width:60px; flex-shrink:0; letter-spacing:.04em; }
.strk { flex:1; height:4px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden; }
.sfil { height:100%; border-radius:2px; }
.spct { font-size:10px; color:#38383f; width:36px; text-align:right; flex-shrink:0;
        font-family:'JetBrains Mono',monospace; }

/* ─── Rec items ──────────────────────────────────────────── */
.ritem { display:flex; gap:16px; padding:16px 0; border-bottom:1px solid rgba(255,255,255,.04); }
.ritem:first-child { padding-top:0; }
.ritem:last-child  { border-bottom:none; padding-bottom:0; }
.rnum {
  width:28px; height:28px; border-radius:50%;
  background: rgba(255,255,255,.04);
  display:flex; align-items:center; justify-content:center;
  font-size:10px; font-weight:700; color:#38383f;
  flex-shrink:0; margin-top:1px;
  font-family:'JetBrains Mono',monospace;
  border: 1px solid rgba(255,255,255,.06);
}
.rtitle { font-size:13px; font-weight:500; color:#d8d8d5; line-height:1.4; margin-bottom:5px; }
.rrat   { font-size:12px; color:#48484f; line-height:1.7; margin-bottom:8px; }
.rimp {
  display:grid; grid-template-columns:1fr 1fr 1fr;
  gap:8px; margin-top:10px; padding-top:10px;
  border-top:1px solid rgba(255,255,255,.04);
}
.rimp-lbl { font-size:9px; color:#2e2e35; letter-spacing:.12em; text-transform:uppercase; margin-bottom:3px; }
.rimp-val { font-size:11px; color:#48484f; }

/* ─── Briefing ───────────────────────────────────────────── */
.brief {
  background: #08080a; border:1px solid rgba(255,255,255,.07);
  border-radius:10px; padding:22px 26px;
  font-size:13px; color:#60606a; line-height:2;
  white-space:pre-wrap; font-family:'Inter',sans-serif;
}
.bsec { font-size:9px; font-weight:700; letter-spacing:.18em; color:#76b900; }

/* ─── Q&A ────────────────────────────────────────────────── */
.qabox {
  background:#08080a; border:1px solid rgba(255,255,255,.07);
  border-radius:10px; padding:20px 24px;
  font-size:13px; color:#60606a; line-height:2; margin-top:14px;
}
.qatag { font-size:9px; font-weight:700; letter-spacing:.16em; color:#76b900; margin-bottom:10px; }

/* ─── Horizon labels ─────────────────────────────────────── */
.hz-s { font-size:9px; font-weight:700; letter-spacing:.12em; color:#3b82f6; }
.hz-m { font-size:9px; font-weight:700; letter-spacing:.12em; color:#9b59b6; }
.hz-l { font-size:9px; font-weight:700; letter-spacing:.12em; color:#76b900; }

/* ─── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:transparent !important;
  border-bottom:1px solid rgba(255,255,255,.07) !important;
  gap:0 !important; margin-bottom:18px !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:#38383f !important;
  font-size:11px !important; font-weight:700 !important;
  padding:8px 24px !important; border-radius:0 !important;
  letter-spacing:.1em !important; text-transform:uppercase !important;
}
.stTabs [aria-selected="true"] {
  background:transparent !important; color:#76b900 !important;
  border-bottom:2px solid #76b900 !important;
}

/* ─── Expanders ──────────────────────────────────────────── */
div[data-testid="stExpander"] {
  background:#0e0e11 !important;
  border:1px solid rgba(255,255,255,.07) !important;
  border-radius:10px !important; margin-bottom:8px !important;
}
details summary { color:#c0c0bd !important; font-size:13px !important; font-weight:500 !important; }

/* ─── Inputs / selects ───────────────────────────────────── */
div[data-testid="stTextInput"] input {
  background:#0e0e11 !important; border-color:rgba(255,255,255,.1) !important;
  color:#e0e0de !important; border-radius:8px !important;
  font-size:12px !important; font-family:'Inter',sans-serif !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color:rgba(118,185,0,.4) !important;
  box-shadow: 0 0 0 2px rgba(118,185,0,.08) !important;
}
div[data-testid="stSelectbox"] > div > div {
  background:#0e0e11 !important; border-color:rgba(255,255,255,.1) !important;
  border-radius:8px !important; color:#c0c0bd !important;
}

/* ─── Buttons (main content) ─────────────────────────────── */
div[data-testid="stButton"] > button {
  background:#0e0e11 !important; border:1px solid rgba(255,255,255,.1) !important;
  color:#60606a !important; border-radius:8px !important;
  font-size:11px !important; font-weight:600 !important;
  letter-spacing:.05em !important; transition:all .15s !important;
}
div[data-testid="stButton"] > button:hover {
  border-color:rgba(118,185,0,.4) !important; color:#76b900 !important;
  background:rgba(118,185,0,.05) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background:rgba(118,185,0,.09) !important;
  border-color:rgba(118,185,0,.3) !important; color:#76b900 !important;
}

/* ─── Alerts ─────────────────────────────────────────────── */
div[data-testid="stInfo"]    { background:rgba(118,185,0,.05) !important; border-color:rgba(118,185,0,.15) !important; color:#76b900 !important; border-radius:8px !important; }
div[data-testid="stSuccess"] { background:rgba(118,185,0,.05) !important; border-color:rgba(118,185,0,.15) !important; color:#76b900 !important; border-radius:8px !important; }
div[data-testid="stError"]   { background:rgba(231,76,60,.05) !important; border-color:rgba(231,76,60,.15) !important; border-radius:8px !important; }
div[data-testid="stWarning"] { background:rgba(230,126,34,.05)!important; border-color:rgba(230,126,34,.15)!important; border-radius:8px !important; }

/* ─── Misc ───────────────────────────────────────────────── */
hr { border-color:rgba(255,255,255,.06) !important; margin:20px 0 !important; }
div[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.07) !important; border-radius:10px !important; }
textarea { background:#08080a !important; border-color:rgba(255,255,255,.1) !important;
           color:#60606a !important; border-radius:8px !important;
           font-size:13px !important; line-height:1.9 !important;
           font-family:'Inter',sans-serif !important; }
div[data-testid="stSpinner"] p { color:#76b900 !important; }

/* ─── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1); border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def _plot(h=None, t=32, b=14, l=8, r=8):
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#48484f", size=10, family="Inter"),
        margin=dict(t=t, b=b, l=l, r=r),
    )
    if h: d["height"] = h
    return d

def _link(ev, lk):
    el = ev.lower()
    for tk, info in lk.items():
        w = [x for x in tk.split() if len(x)>4]
        if w and sum(1 for x in w if x in el) >= min(2, len(w)):
            return info["url"], info["source"]
    return "", ""

def _evhtml(evs, lk):
    out = ""
    for e in evs:
        url, src = _link(e, lk)
        snip = ". ".join(e.split(". ")[:2]) + ("." if len(e.split(". "))>2 else "")
        a = (f' <a href="{url}" target="_blank">↗ {src}</a>' if url else "")
        out += f'<div class="ev">{snip}{a}</div>'
    return out

def answer_ceo_question(q):
    try:
        import ollama
        from processor.knowledge_store import query as kb_query
        try:
            from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P
        except Exception:
            LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P = "llama3.1:8b", 0.1, 0.9
        chunks = kb_query(q, top_k=8)
        if not chunks: return "Knowledge base empty. Run python pipeline.py first."
        ctx = "\n\n".join([f"[{c['source']} | {c['date']}]\n{c['text']}" for c in chunks])
        extra = ""
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH) as f: r = json.load(f)
            recs = r.get("recommendations",[])[:3]
            if recs:
                extra = "\n\nTop recommendations:\n"+"\n".join([f"- {x.get('recommendation','')}" for x in recs])
        sys_p = (f"You are a strategic advisor to the CEO of {COMPANY_NAME}. "
                 "Answer with evidence-based reasoning. Concise (150-200 words). "
                 "End with: Sources: [names]")
        resp = ollama.chat(model=LLM_MODEL,
            options={"temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P, "num_predict": 500},
            messages=[{"role":"system","content":sys_p},
                      {"role":"user","content":f"CONTEXT:\n{ctx}{extra}\n\nQUESTION: {q}"}])
        return resp["message"]["content"]
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "refused" in msg: return "Ollama not running. Run: ollama serve"
        if "not found" in msg.lower(): return "Model missing. Run: ollama pull llama3.1:8b"
        return f"Error: {msg}"


# ═══════════════════════════════════════════════════════════════
#  LOAD REPORT + NAV STATE
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_report():
    if not os.path.exists(REPORT_PATH): return {}
    with open(REPORT_PATH) as f: return json.load(f)

if "page" not in st.session_state: st.session_state.page = "overview"

NAV = [
    ("overview",     "Overview"),
    ("feed",         "Market feed"),
    ("sentiment",    "Sentiment"),
    ("intelligence", "Intelligence"),
    ("recs",         "Recommendations"),
    ("briefing",     "CEO briefing"),
    ("qa",           "Ask the advisor"),
]

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 20px 20px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:8px">
      <div style="font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#76b900;margin-bottom:5px">NVDA · SEMICONDUCTORS</div>
      <div style="font-size:15px;font-weight:600;color:#f2f2f0;letter-spacing:-.2px">CEO Intelligence</div>
    </div>
    <div style="font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#28282f;padding:14px 20px 7px">Monitor</div>
    """, unsafe_allow_html=True)

    for key, label in NAV[:3]:
        active = st.session_state.page == key
        prefix = "● " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nb_{key}", use_container_width=True):
            st.session_state.page = key; st.rerun()

    st.markdown('<div style="font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#28282f;padding:14px 20px 7px">Analysis</div>', unsafe_allow_html=True)

    for key, label in NAV[3:5]:
        active = st.session_state.page == key
        prefix = "● " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nb_{key}", use_container_width=True):
            st.session_state.page = key; st.rerun()

    st.markdown('<div style="font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#28282f;padding:14px 20px 7px">Briefing</div>', unsafe_allow_html=True)

    for key, label in NAV[5:]:
        active = st.session_state.page == key
        prefix = "● " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nb_{key}", use_container_width=True):
            st.session_state.page = key; st.rerun()

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:20px 0 16px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#28282f;padding:0 20px 7px">Pipeline</div>', unsafe_allow_html=True)

    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    if st.button("▶  Run pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running pipeline…"):
            os.system(f'python "{os.path.join(_ROOT,"pipeline.py")}"')
        st.cache_data.clear(); st.rerun()
    if st.button("⟳  Force re-collect", use_container_width=True):
        with st.spinner("Re-collecting…"):
            os.system(f'python "{os.path.join(_ROOT,"pipeline.py")}" --force')
        st.cache_data.clear(); st.rerun()


# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════
report    = load_report()
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
lk        = {d.get("title","").lower()[:70]: {"url":d.get("url",""),"source":d.get("source","")}
             for d in docs_meta if d.get("url")}

page = st.session_state.page
gen_at   = meta.get("generated_at","N/A")[:16]
n_docs   = meta.get("total_documents",0)
n_chunks = meta.get("total_chunks",0)
n_src    = meta.get("num_sources",0)
ws       = sent.get("weighted_score",0)
wslbl    = "BULLISH" if ws>.05 else ("BEARISH" if ws<-.05 else "NEUTRAL")

PAGE_META = {
    "overview":     ("OVERVIEW",     "Company snapshot"),
    "feed":         ("MARKET FEED",  "Live articles · 8 sources"),
    "sentiment":    ("SENTIMENT",    "FinBERT + RoBERTa dual-model analysis"),
    "intelligence": ("INTELLIGENCE", "RAG-powered opportunity & risk detection"),
    "recs":         ("STRATEGY",     "CEO-level recommendations · priority sorted"),
    "briefing":     ("BRIEFING",     "Executive morning summary"),
    "qa":           ("ADVISOR",      "Ask anything · Llama 3.1:8b + ChromaDB"),
}
eyebrow, subtitle = PAGE_META.get(page, ("DASHBOARD",""))

# ─── page header ───────────────────────────────────────────────
st.markdown(f"""
<div class="ph">
  <div class="ph-left">
    <div class="ph-eyebrow">NVIDIA · {eyebrow}</div>
    <div class="ph-title">{subtitle}</div>
  </div>
  <div class="ph-right">
    <div class="ph-stat">
      <div class="ph-stat-val">{n_docs:,}</div>
      <div class="ph-stat-lbl">Articles</div>
    </div>
    <div class="ph-divider"></div>
    <div class="ph-stat">
      <div class="ph-stat-val">{n_chunks:,}</div>
      <div class="ph-stat-lbl">Chunks</div>
    </div>
    <div class="ph-divider"></div>
    <div class="ph-stat">
      <div class="ph-stat-val" style="font-size:14px;color:{'#76b900' if ws>0 else '#e74c3c'}">{ws:+.3f}</div>
      <div class="ph-stat-lbl">Sentiment</div>
    </div>
    <div class="ph-divider"></div>
    <div class="status-badge"><div class="status-dot"></div>Live · {gen_at}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Articles collected", f"{n_docs:,}",   help="Raw documents from all 8 sources")
    c2.metric("ChromaDB chunks",    f"{n_chunks:,}",  help="400-char overlapping chunks for vector retrieval")
    c3.metric("Active sources",     f"{n_src}",       help="Unique data sources contributing documents")
    c4.metric("Weighted sentiment", f"{ws:+.3f}",     help=f"{wslbl} · range −1 to +1")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        if opps:
            o=opps[0]; conf=o.get("confidence_score",.5); imp=o.get("impact_level","Medium")
            ev_h = _evhtml(o.get("evidence",[])[:2], lk)
            st.markdown(f"""
            <div class="gcard">
              <div class="gcard-title">Top opportunity</div>
              <div class="irow" style="padding-top:0">
                <div class="irow-head">
                  <div class="irow-name">{o.get('title','')}</div>
                  <span class="gbd gbd-g">{imp}</span>
                </div>
                <div class="irow-desc">{o.get('description','')}</div>
                {ev_h}
                <div class="cbar-wrap" style="margin-top:10px">
                  <div class="cbar-track"><div class="cbar-fill" style="width:{int(conf*100)}%;background:#76b900"></div></div>
                  <div class="cbar-pct">{conf:.0%}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        if risks:
            r=risks[0]; conf=r.get("confidence_score",.5); sev=r.get("severity","High")
            sc={"Critical":"#e74c3c","High":"#e67e22","Medium":"#c9a600","Low":"#76b900"}.get(sev,"#888")
            bc={"Critical":"gbd-r","High":"gbd-o","Medium":"gbd-y","Low":"gbd-g"}.get(sev,"gbd-o")
            ev_h=_evhtml(r.get("evidence",[])[:2], lk)
            st.markdown(f"""
            <div class="gcard">
              <div class="gcard-title">Top risk</div>
              <div class="irow" style="padding-top:0">
                <div class="irow-head">
                  <div class="irow-name">{r.get('title','')}</div>
                  <span class="gbd {bc}">{sev}</span>
                </div>
                <div class="irow-desc">{r.get('description','')}</div>
                {ev_h}
                <div class="cbar-wrap" style="margin-top:10px">
                  <div class="cbar-track"><div class="cbar-fill" style="width:{int(conf*100)}%;background:{sc}"></div></div>
                  <div class="cbar-pct">{conf:.0%}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    if docs_meta:
        rows=""
        for d in docs_meta[:10]:
            s=d.get("sentiment","neutral")
            dot={"positive":"#76b900","negative":"#e74c3c","neutral":"#28282f"}.get(s,"#28282f")
            url=d.get("url",""); ttl=d.get("title","")[:92]
            lo=(f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit">' if url else "")
            lc="</a>" if url else ""
            rows+=f"""<div class="nrow">
              <div class="ndot" style="background:{dot}"></div>
              <div><div class="ntitle">{lo}{ttl}{lc}</div>
              <div class="nmeta">{d.get('source','')} · {d.get('date','')[:10]}</div></div></div>"""
        st.markdown(f'<div class="gcard"><div class="gcard-title">Recent articles</div>{rows}</div>',
                    unsafe_allow_html=True)

    if meta.get("sources"):
        with st.expander("Active data sources"):
            cols=st.columns(3)
            for i,s in enumerate(sorted(meta["sources"])):
                cols[i%3].markdown(f'<span style="font-size:11px;color:#48484f">✓  {s}</span>',
                                   unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MARKET FEED
# ═══════════════════════════════════════════════════════════════
elif page == "feed":
    if not docs_meta:
        st.info("Run the pipeline to populate the market feed.")
    else:
        cf1,cf2,cf3 = st.columns([3,1,1])
        with cf1:
            srch=st.text_input("",placeholder="Semantic search — e.g. AMD, export controls, Blackwell…",
                               label_visibility="collapsed")
        with cf2:
            sf=st.selectbox("",["All sentiments","positive","neutral","negative"],
                            label_visibility="collapsed")
        with cf3:
            all_src=sorted({d.get("source","") for d in docs_meta if d.get("source")})
            srcf=st.selectbox("",["All sources"]+all_src,label_visibility="collapsed")

        filtered=docs_meta
        if srch and len(srch.strip())>2:
            try:
                from processor.knowledge_store import query as kb_query
                sem=kb_query(srch.strip(),top_k=15)
                stit={r["title"].lower()[:80] for r in sem if r.get("title")}
                surl={r["url"] for r in sem if r.get("url")}
                matched=[d for d in docs_meta if d.get("title","").lower()[:80] in stit or d.get("url","") in surl]
                filtered=matched if matched else docs_meta
                if matched: st.success(f"{len(matched)} articles matched for \"{srch}\"")
            except Exception as ex:
                filtered=[d for d in docs_meta if srch.lower() in d.get("title","").lower()]
                st.caption(f"Semantic search unavailable — keyword fallback ({ex})")
        if sf!="All sentiments": filtered=[d for d in filtered if d.get("sentiment")==sf]
        if srcf!="All sources":  filtered=[d for d in filtered if d.get("source")==srcf]

        rows=""
        for d in filtered[:30]:
            s=d.get("sentiment","neutral")
            dot={"positive":"#76b900","negative":"#e74c3c","neutral":"#28282f"}.get(s,"#28282f")
            sc=d.get("sentiment_score",0)
            url=d.get("url",""); ttl=d.get("title","")[:98]
            lo=(f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit">' if url else "")
            lc="</a>" if url else ""
            rows+=f"""<div class="nrow">
              <div class="ndot" style="background:{dot}"></div>
              <div><div class="ntitle">{lo}{ttl}{lc}</div>
              <div class="nmeta">{d.get('source','')} · {d.get('date','')[:10]} · {sc:.2f}</div></div></div>"""
        st.markdown(
            f'<div class="gcard"><div class="gcard-title">{min(30,len(filtered))} of {len(filtered)} articles</div>'
            f'{rows}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SENTIMENT
# ═══════════════════════════════════════════════════════════════
elif page == "sentiment":
    if not sent:
        st.info("No sentiment data. Run the pipeline first.")
    else:
        ns=sent.get("news_sentiment",{}); ps=sent.get("public_sentiment",{})
        n_avg=ns.get("avg_score",0.0);    p_avg=ps.get("avg_score",0.0)

        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Overall mood",   wslbl)
        k2.metric("Positive",       f"{sent.get('positive_pct',0)}%", sent.get("positive",0))
        k3.metric("Neutral",        f"{sent.get('neutral_pct',0)}%",  sent.get("neutral",0))
        k4.metric("Negative",       f"{sent.get('negative_pct',0)}%", sent.get("negative",0))
        k5.metric("Weighted score", f"{ws:+.3f}", wslbl)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        cl,cr=st.columns(2,gap="medium")

        with cl:
            fd=go.Figure(go.Pie(
                labels=["Positive","Neutral","Negative"],
                values=[sent.get("positive",0),sent.get("neutral",0),sent.get("negative",0)],
                hole=.65, marker_colors=["#76b900","#1a1a1f","#e74c3c"],
                textinfo="label+percent", textfont=dict(color="#60606a",size=10),
            ))
            fd.update_layout(title=dict(text="Sentiment split",font=dict(color="#38383f",size=11)),
                             showlegend=False,**_plot(270))
            st.plotly_chart(fd,use_container_width=True)

        with cr:
            bc="#76b900" if ws>0 else "#e74c3c"
            fg=go.Figure(go.Indicator(
                mode="gauge+number", value=ws,
                number={"valueformat":"+.3f","font":{"color":bc,"size":26,"family":"JetBrains Mono"}},
                gauge={
                    "axis":{"range":[-1,1],"tickwidth":1,"tickcolor":"#1a1a1f",
                            "tickfont":{"color":"#38383f","size":9}},
                    "bar":{"color":bc,"thickness":.2},
                    "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(255,255,255,.06)",
                    "steps":[{"range":[-1,-.3],"color":"rgba(231,76,60,.06)"},
                             {"range":[-.3,.3],"color":"rgba(255,255,255,.015)"},
                             {"range":[.3,1],"color":"rgba(118,185,0,.06)"}],
                    "threshold":{"line":{"color":"rgba(255,255,255,.12)","width":1.5},"thickness":.75,"value":0},
                },
                title={"text":f"Weighted score<br><sup>{wslbl}</sup>","font":{"color":"#38383f","size":11}},
            ))
            fg.update_layout(**_plot(270,t=60,l=30,r=30))
            st.plotly_chart(fg,use_container_width=True)

        if docs_meta:
            ss=defaultdict(lambda:{"positive":0,"neutral":0,"negative":0})
            for d in docs_meta: ss[d.get("source","?")[:20]][d.get("sentiment","neutral")]+=1
            si=sorted(ss.items(),key=lambda x:sum(x[1].values()),reverse=True)[:10]
            src_=[s for s,_ in si]
            fb=go.Figure()
            fb.add_trace(go.Bar(name="Positive",x=src_,y=[v["positive"] for _,v in si],marker_color="#76b900"))
            fb.add_trace(go.Bar(name="Neutral", x=src_,y=[v["neutral"]  for _,v in si],marker_color="#1a1a1f"))
            fb.add_trace(go.Bar(name="Negative",x=src_,y=[v["negative"] for _,v in si],marker_color="#e74c3c"))
            fb.update_layout(
                title=dict(text="Sentiment by source",font=dict(color="#38383f",size=11)),
                barmode="stack",
                legend=dict(orientation="h",y=-0.28,font=dict(color="#48484f",size=9)),
                xaxis=dict(tickangle=-38,tickfont=dict(color="#38383f",size=9),gridcolor="rgba(255,255,255,.03)"),
                yaxis=dict(tickfont=dict(color="#38383f",size=9),gridcolor="rgba(255,255,255,.03)"),
                **_plot(290))
            st.plotly_chart(fb,use_container_width=True)

        cl2,cr2=st.columns(2,gap="medium")
        with cl2:
            if ps.get("count",0)>0:
                fc=go.Figure(go.Bar(
                    x=["News (FinBERT)","Public (RoBERTa)"], y=[n_avg,p_avg],
                    marker_color=["#76b900" if n_avg>=0 else "#e74c3c",
                                  "#3b82f6" if p_avg>=0 else "#e74c3c"],
                    text=[f"{n_avg:+.3f}",f"{p_avg:+.3f}"], textposition="outside",
                    textfont=dict(color="#60606a",size=10),
                ))
                fc.add_hline(y=0,line_dash="dot",line_color="rgba(255,255,255,.06)",line_width=1)
                fc.update_layout(
                    title=dict(text="News vs public score",font=dict(color="#38383f",size=11)),
                    xaxis=dict(tickfont=dict(color="#48484f",size=10),gridcolor="rgba(255,255,255,.03)"),
                    yaxis=dict(range=[-1.2,1.2],tickfont=dict(color="#38383f",size=9),
                               gridcolor="rgba(255,255,255,.03)"),
                    **_plot(250))
                st.plotly_chart(fc,use_container_width=True)
        with cr2:
            mb=sent.get("model_breakdown",{}); fb_n=mb.get("FinBERT",0); rb_n=mb.get("RoBERTa",0)
            tot=fb_n+rb_n or 1
            st.markdown(f"""
            <div class="gcard" style="margin-top:0">
              <div class="gcard-title">Model usage</div>
              <div class="srow"><div class="slbl">FinBERT</div>
                <div class="strk"><div class="sfil" style="width:{fb_n/tot*100:.0f}%;background:#3b82f6"></div></div>
                <div class="spct">{fb_n}</div></div>
              <div class="srow"><div class="slbl">RoBERTa</div>
                <div class="strk"><div class="sfil" style="width:{rb_n/tot*100:.0f}%;background:#9b59b6"></div></div>
                <div class="spct">{rb_n}</div></div>
              <div style="font-size:10px;color:#2e2e35;margin-top:14px;line-height:1.8">
                FinBERT → financial sources (Yahoo Finance, NVIDIA IR)<br>
                RoBERTa → social sources (Reddit, HackerNews)
              </div>
            </div>""", unsafe_allow_html=True)

        ts=sent.get("trend_series",[])
        if ts and len(ts)>=2:
            dates=[p["date"] for p in ts]; scores=[p["score"] for p in ts]
            labels=[p["label"] for p in ts]
            mc=["#76b900" if l=="positive" else ("#e74c3c" if l=="negative" else "#28282f") for l in labels]
            ft=go.Figure()
            ft.add_hrect(y0=-.05,y1=.05,fillcolor="rgba(255,255,255,.01)",line_width=0)
            ft.add_trace(go.Scatter(
                x=dates, y=scores, mode="lines+markers",
                line=dict(color="#76b900",width=1.8),
                marker=dict(size=6,color=mc,line=dict(width=1.5,color="#08080a")),
                fill="tozeroy", fillcolor="rgba(118,185,0,.04)",
                hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
            ))
            if len(scores)>=4:
                mom=(sum(scores[-2:])/2)-(sum(scores[:2])/2)
                mt="↑ improving" if mom>.02 else ("↓ declining" if mom<-.02 else "→ stable")
                ft.add_annotation(x=dates[-1],y=scores[-1],text=mt,showarrow=True,arrowhead=2,
                                  font=dict(size=10,color="#76b900"),arrowcolor="#76b900")
            ft.update_layout(
                title=dict(text="30-day sentiment trend",font=dict(color="#38383f",size=11)),
                xaxis=dict(tickangle=-30,tickfont=dict(color="#38383f",size=9),gridcolor="rgba(255,255,255,.03)"),
                yaxis=dict(range=[-1.1,1.1],tickfont=dict(color="#38383f",size=9),gridcolor="rgba(255,255,255,.03)"),
                hovermode="x unified",**_plot(290))
            st.plotly_chart(ft,use_container_width=True)

            t1,t2,t3,t4=st.columns(4)
            t1.metric("Data points",len(scores))
            t2.metric("Mean score", f"{float(np.mean(scores)):+.3f}")
            t3.metric("Volatility", f"{float(np.std(scores)):.3f}",help="Std dev — higher = more swings")
            t4.metric("Latest",     f"{scores[-1]:+.3f}",
                      f"{'↑' if scores[-1]>scores[-2] else '↓'} vs prev" if len(scores)>=2 else "")
        else:
            st.info("Trend chart needs ≥2 dates. Re-run on separate days.")


# ═══════════════════════════════════════════════════════════════
#  INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif page == "intelligence":
    t1,t2,t3=st.tabs(["  Opportunities  ","  Risks  ","  Trends  "])

    with t1:
        if opps:
            rows=""
            for o in opps:
                conf=o.get("confidence_score",.5); imp=o.get("impact_level","Medium")
                ev=_evhtml(o.get("evidence",[]),lk)
                rows+=f"""<div class="irow">
                  <div class="irow-head"><div class="irow-name">{o.get('title','')}</div>
                    <span class="gbd gbd-g">{imp}</span></div>
                  <div class="irow-desc">{o.get('description','')}</div>{ev}
                  <div class="cbar-wrap" style="margin-top:9px">
                    <div class="cbar-track"><div class="cbar-fill" style="width:{int(conf*100)}%;background:#76b900"></div></div>
                    <div class="cbar-pct">{conf:.0%} · {o.get('category','')}</div></div></div>"""
            st.markdown(f'<div class="gcard">{rows}</div>',unsafe_allow_html=True)
        else: st.info("No opportunities. Run the pipeline.")

    with t2:
        if risks:
            SC={"Critical":"#e74c3c","High":"#e67e22","Medium":"#c9a600","Low":"#76b900"}
            BC={"Critical":"gbd-r","High":"gbd-o","Medium":"gbd-y","Low":"gbd-g"}
            rows=""
            for r in risks:
                conf=r.get("confidence_score",.5); sev=r.get("severity","High")
                sc=SC.get(sev,"#888"); bc=BC.get(sev,"gbd-o")
                ev=_evhtml(r.get("evidence",[]),lk)
                rows+=f"""<div class="irow">
                  <div class="irow-head"><div class="irow-name">{r.get('title','')}</div>
                    <span class="gbd {bc}">{sev}</span></div>
                  <div class="irow-desc">{r.get('description','')}</div>{ev}
                  <div class="cbar-wrap" style="margin-top:9px">
                    <div class="cbar-track"><div class="cbar-fill" style="width:{int(conf*100)}%;background:{sc}"></div></div>
                    <div class="cbar-pct">{conf:.0%} · {r.get('category','')}</div></div></div>"""
            st.markdown(f'<div class="gcard">{rows}</div>',unsafe_allow_html=True)
        else: st.info("No risks. Run the pipeline.")

    with t3:
        if trends:
            rows=""
            for t in trends:
                h=t.get("time_horizon","")
                hc="hz-s" if "Short" in h else ("hz-l" if "Long" in h else "hz-m")
                hl="SHORT-TERM" if "Short" in h else ("LONG-TERM" if "Long" in h else "MID-TERM")
                ev=_evhtml(t.get("evidence",[]),lk)
                rows+=f"""<div class="irow">
                  <div class="irow-head">
                    <div><div class="{hc}" style="margin-bottom:3px">{hl}</div>
                      <div class="irow-name">{t.get('title','')}</div></div>
                    <span class="gbd gbd-b">{t.get('relevance','Medium')}</span></div>
                  <div class="irow-desc">{t.get('description','')}</div>{ev}
                  <div class="irow-cat">{t.get('category','')}</div></div>"""
            st.markdown(f'<div class="gcard">{rows}</div>',unsafe_allow_html=True)
        else: st.info("No trends. Run the pipeline.")


# ═══════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "recs":
    if not recs:
        st.info("No recommendations. Run the pipeline.")
    else:
        hc=sum(1 for r in recs if r.get("priority")=="High")
        mc_=sum(1 for r in recs if r.get("priority")=="Medium")
        lc=sum(1 for r in recs if r.get("priority")=="Low")
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:18px">
          <span class="gbd gbd-r">{hc} High</span>
          <span class="gbd gbd-o">{mc_} Medium</span>
          <span class="gbd gbd-g">{lc} Low</span>
        </div>""",unsafe_allow_html=True)

        rows=""
        for i,rec in enumerate(recs,1):
            pri=rec.get("priority","Medium")
            pc={"High":"gbd-r","Medium":"gbd-o","Low":"gbd-g"}.get(pri,"gbd-o")
            rl=rec.get("risk_level","Medium")
            rc={"High":"gbd-r","Medium":"gbd-o","Low":"gbd-g"}.get(rl,"gbd-o")
            imp=rec.get("expected_impact",{}); ra=rec.get("risk_assessment",{})
            ev=_evhtml(rec.get("supporting_evidence",[]),lk)
            rows+=f"""<div class="ritem">
              <div class="rnum">{i:02d}</div>
              <div style="flex:1">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:6px">
                  <div class="rtitle">{rec.get('recommendation','')}</div>
                  <span class="gbd {pc}" style="flex-shrink:0">{pri}</span></div>
                <div class="rrat">{rec.get('rationale','')}</div>
                {ev}
                <div class="rimp">
                  <div><div class="rimp-lbl">Revenue</div><div class="rimp-val">{imp.get('revenue','N/A')}</div></div>
                  <div><div class="rimp-lbl">Timeline</div><div class="rimp-val">{imp.get('timeline','N/A')}</div></div>
                  <div><div class="rimp-lbl">Risk</div><span class="gbd {rc}">{rl}</span></div>
                </div>
              </div></div>"""
        st.markdown(f'<div class="gcard">{rows}</div>',unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CEO BRIEFING
# ═══════════════════════════════════════════════════════════════
elif page == "briefing":
    col_b,col_t=st.columns([3,1],gap="medium")

    with col_b:
        if briefing:
            fmt=briefing
            for s in ["WHAT HAPPENED","WHY IT MATTERS","WHAT TO DO NEXT"]:
                fmt=fmt.replace(s,f'<span class="bsec">{s}</span>')
            st.markdown(f'<div class="gcard"><div class="gcard-title">Morning briefing</div>'
                        f'<div class="brief">{fmt}</div></div>',unsafe_allow_html=True)
        else:
            st.info("No briefing. Run the pipeline.")

    with col_t:
        if trends:
            rows=""
            for t in trends:
                h=t.get("time_horizon","")
                hc="hz-s" if "Short" in h else ("hz-l" if "Long" in h else "hz-m")
                hl="SHORT" if "Short" in h else ("LONG" if "Long" in h else "MID")
                rows+=f"""<div class="irow">
                  <div class="{hc}" style="margin-bottom:3px">{hl}-TERM</div>
                  <div class="irow-name" style="font-size:12px">{t.get('title','')}</div>
                  <div class="irow-desc" style="font-size:11px;margin-top:3px">{t.get('description','')[:110]}…</div>
                </div>"""
            st.markdown(f'<div class="gcard"><div class="gcard-title">Emerging trends</div>{rows}</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  Q&A
# ═══════════════════════════════════════════════════════════════
elif page == "qa":
    suggested=[
        "What are AMD and Intel doing in AI chips?",
        "What is NVIDIA's biggest risk right now?",
        "Should NVIDIA expand into automotive AI?",
        "What does sentiment say about investor confidence?",
        "What should NVIDIA prioritize this quarter?",
    ]
    sq=st.columns(3)
    for i,s_ in enumerate(suggested):
        if sq[i%3].button(s_,key=f"sq{i}"):
            st.session_state["ceo_q"]=s_

    question=st.text_input("",value=st.session_state.get("ceo_q",""),
                           placeholder="Ask a strategic question — e.g. What is NVIDIA's biggest risk?",
                           label_visibility="collapsed")

    if st.button("Get AI answer",type="primary") and question:
        with st.spinner("Searching ChromaDB · reasoning with Llama 3.1…"):
            ans=answer_ceo_question(question)
        st.markdown(f'<div class="qabox"><div class="qatag">AI advisor · RAG-grounded</div>{ans}</div>',
                    unsafe_allow_html=True)
        if "ceo_q" in st.session_state: del st.session_state["ceo_q"]

    st.markdown('<div style="height:20px"></div>',unsafe_allow_html=True)
    h1,h2,h3=st.columns(3,gap="medium")
    for col,num,head,desc in [
        (h1,"01","Query ChromaDB","Top 8 chunks retrieved by cosine similarity across all stored vectors"),
        (h2,"02","Build context","Chunks + top 3 recommendations injected into Llama 3.1:8b prompt"),
        (h3,"03","Generate answer","150–200 word grounded response · temperature 0.1 · source attribution"),
    ]:
        col.markdown(f"""
        <div class="gcard" style="margin-top:0">
          <div style="font-size:32px;font-weight:700;color:#1a1a1f;margin-bottom:10px;
               font-family:'JetBrains Mono',monospace;letter-spacing:-1px">{num}</div>
          <div style="font-size:12px;font-weight:600;color:#60606a;margin-bottom:6px">{head}</div>
          <div style="font-size:11px;color:#38383f;line-height:1.7">{desc}</div>
        </div>""",unsafe_allow_html=True)

