"""
Margin — ask your documents.

Attach a PDF, Word, Markdown or text file in the message bar. Margin detects what kind of
document it is (plain narrative, long structured report, or visual with charts/tables/footnotes),
reads it with the matching RAG strategy, and answers with citations. Pip, the mascot, narrates.

Pipeline (rag_engine.py):
Document -> Parsing -> Chunking -> Embedding -> Retrieval -> Reranking -> Context -> LLM -> Evaluation
"""

import datetime as dt
import hashlib
import html
import os
import random
import re
import time
import uuid

import numpy as np
import pandas as pd
import streamlit as st

import mascot
import rag_engine as rag
import usage as usage_mod

CHAT_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
TYPE_SHORT = {"narrative": "Narrative", "structured": "Structured", "visual": "Visual"}
TYPE_PLAIN = {"narrative": "a plain narrative report", "structured": "a long, structured report",
              "visual": "a visual report with charts, tables and footnotes"}
TYPE_PLAN = {"narrative": "I'll read it page by page and keep each fact with its sentence.",
             "structured": "I'll map its chapters and sections first, so answers stay in the right place.",
             "visual": "I'll read the text, then study each page image so no chart number is missed."}
KIND = {"text": "text", "table": "table", "visual": "chart/table from page image", "footnote": "endnote"}
ICONS = {
    "narrative": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10 13h6M10 17h6"/>',
    "structured": '<path d="M12 3 3 8l9 5 9-5-9-5z"/><path d="m3 12.5 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "visual": '<path d="M4 20V11M10 20V5M16 20v-7M21 20H3"/>',
}


def icon(name):
    return (f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


st.set_page_config(page_title="Margin · Ask your documents", page_icon="🔮", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Visual design: lavender canvas, white rounded panel, black primary actions (Cortex-inspired)
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root{--ink:#16131F;--muted:#6F6A82;--faint:#A19CB3;--line:#ECE8F5;--soft:#F6F3FD;--violet:#8B5CF6;--violet-ink:#6D28D9;
  --lilac:linear-gradient(90deg,#C4B5FD 0%,#A78BFA 45%,#8B5CF6 100%)}
.stApp{background:radial-gradient(1100px 600px at 12% 0%,#E6DCF7 0%,transparent 60%),
  radial-gradient(900px 600px at 95% 100%,#F3E3F2 0%,transparent 60%),#EEE9F6;color:var(--ink)}
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button p,.stApp span,.stApp div{
  font-family:'Plus Jakarta Sans',system-ui,-apple-system,sans-serif}
[data-testid="stIconMaterial"],.material-symbols-rounded{font-family:'Material Symbols Rounded'!important}
[data-testid="stHeader"]{background:transparent}
.stMain .block-container{background:#fff;border-radius:28px 28px 0 0;max-width:none;margin:14px 14px 0 0;
  padding:3.1rem 2.2rem 2rem 2.2rem;min-height:calc(100vh - 112px);flex:1 0 auto!important}
.stMain>div:not(.block-container):not([data-testid="stBottom"]){background:#fff;margin-right:14px}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:transparent!important;border-right:0}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:1.2rem}
.st-key-center,.st-key-center_live{max-width:840px;margin:0 auto}
[data-testid="stBottom"]{background:transparent!important;padding:0 14px 14px 0}
[data-testid="stBottom"]>div{background:#fff!important;border-radius:0 0 28px 28px}
[data-testid="stBottomBlockContainer"]{background:transparent!important;max-width:840px;padding:.2rem 1rem 1.2rem 1rem}
[data-testid="stChatInput"]{background:#fff;border:1px solid #E6DFFA;border-radius:22px;
  box-shadow:0 12px 34px rgba(124,92,255,.12),0 0 0 4px rgba(196,181,253,.14)}
[data-testid="stChatInput"] textarea{font-size:.98rem}
.stButton>button,.stDownloadButton>button,[data-testid="stPopover"] button{border-radius:12px;border:1px solid var(--line);
  background:#fff;color:var(--ink);font-weight:500;transition:background .15s,border-color .15s}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stPopover"] button:hover{border-color:#D8CCFB;background:var(--soft);color:var(--ink)}
.stButton>button[kind="primary"]{background:#14121C;border:0;color:#fff;font-weight:600}
.stButton>button[kind="primary"]:hover{background:#2A2638;color:#fff}
.stButton>button[kind="tertiary"]{border:0;background:transparent;justify-content:flex-start;padding:.38rem .6rem;color:#2C2838;
  font-weight:500;width:100%;border-radius:10px}
.stButton>button[kind="tertiary"]:hover{background:#F1ECFD;color:var(--ink)}
.stButton>button:focus-visible{outline:2px solid #A78BFA;outline-offset:2px}
.stButton>button p{white-space:normal;text-align:inherit}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.3rem}
.stButton>button[kind="tertiary"]>div{justify-content:flex-start!important;width:100%}
.stButton>button[kind="tertiary"] p{text-align:left}
[data-testid="stSidebar"] .stButton>button[kind="tertiary"] p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
[data-testid="stTextInputRootElement"]{border-radius:12px;background:#fff}
.stTextInput input{border-radius:12px;background:#fff}
[data-testid="stExpander"] details{border:1px solid var(--line);border-radius:14px;background:#fff}
[data-testid="stChatMessage"]{background:transparent;padding:.4rem 0;gap:.8rem}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse;background:#F4F0FE;
  border-radius:20px 20px 6px 20px;padding:.7rem 1rem;margin-left:auto;max-width:78%;width:fit-content}
[data-testid="stChatMessageAvatarUser"]{display:none}
[data-testid="stChatMessageAvatarCustom"] img,[data-testid^="stChatMessageAvatar"] img{border-radius:50%}
.m-brand{display:flex;align-items:center;gap:.55rem;font-weight:700;font-size:1.25rem;color:var(--ink);margin:0 0 .9rem .2rem}
.m-brand .pip-svg{width:34px;height:36px}
.m-group{font-size:.74rem;color:var(--muted);margin:.6rem 0 .6rem .6rem;font-weight:500}
[data-testid="stPopoverBody"]{background:#fff;border:1px solid var(--line);border-radius:16px}
.m-hello{font-size:2.3rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;margin:.2rem 0 0 0;text-align:center;
  background:var(--lilac);-webkit-background-clip:text;background-clip:text;color:transparent}
.m-h1{font-size:2.3rem;font-weight:700;letter-spacing:-.03em;line-height:1.15;margin:.1rem 0 .4rem 0;text-align:center;color:var(--ink)}
.stApp p.m-sub{text-align:center;color:var(--muted);margin:.2rem auto 1.4rem auto!important;max-width:54ch}
.m-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.2rem 0 .6rem 0}
.m-card{border:1px solid var(--line);border-radius:18px;padding:1rem 1.05rem;background:#fff}
.m-card .ic{color:#2C2838;margin-bottom:.9rem;display:block}
.m-card h4{margin:0 0 .25rem 0;font-size:.98rem;font-weight:600;color:var(--ink)}
.m-card p{margin:0;font-size:.85rem;color:var(--faint);line-height:1.45}
.m-hint{text-align:center;color:var(--muted);font-size:.86rem;margin:.6rem 0}
.st-key-sugg .stButton>button{min-height:84px;border-radius:18px;justify-content:flex-start;align-items:flex-start;
  text-align:left;padding:.9rem 1rem;font-weight:500}
.m-badge{display:inline-block;font-size:.72rem;font-weight:600;padding:.15rem .6rem;border-radius:999px;margin-right:.35rem}
.m-badge.narrative{background:#EDE7FE;color:#5B21B6}.m-badge.structured{background:#E0F2FE;color:#0C5A87}
.m-badge.visual{background:#FFE8DA;color:#9A3412}
.m-chip{display:inline-block;font-size:.76rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;
  padding:.1rem .55rem;margin:.12rem .25rem .12rem 0;background:#fff}
.m-doc{border:1px solid var(--line);border-radius:16px;padding:.75rem .9rem;margin:.35rem 0;background:#fff}
.m-doc .t{font-weight:600;color:var(--ink);font-size:.95rem;margin-bottom:.25rem}
.m-doc .n{color:var(--muted);font-size:.8rem}
.m-top{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;min-height:40px}
.m-top .ttl{font-weight:600;font-size:1rem;color:var(--ink);margin-right:.4rem}
.m-usage{border:1px solid var(--line);background:#fff;border-radius:16px;padding:.8rem .9rem;margin:1rem 0 1.2rem 0}
.m-usage .h{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-bottom:.4rem}
.m-usage .big{font-weight:700;font-size:1.05rem;color:var(--ink)}
.m-meter{height:7px;border-radius:99px;background:#F0ECF8;overflow:hidden;margin:.45rem 0 .4rem 0}
.m-meter i{display:block;height:100%;border-radius:99px;background:var(--lilac)}
.m-meter.amber i{background:linear-gradient(90deg,#FCD34D,#F59E0B)}.m-meter.red i{background:linear-gradient(90deg,#FCA5A5,#EF4444)}
.m-usage .s{font-size:.78rem;color:var(--muted);line-height:1.5}
.m-banner{border:1px solid #FDE7C2;background:#FFF8EC;color:#7A4A06;border-radius:14px;padding:.6rem .9rem;font-size:.88rem;margin:.4rem 0}
.m-step{font-weight:600;font-size:1.05rem;margin:1.3rem 0 .25rem 0;color:var(--ink)}
.m-src{border-left:3px solid #C4B5FD;padding:.55rem .9rem;margin:.45rem 0 .7rem 0;background:#FAF8FF;border-radius:0 12px 12px 0;
  line-height:1.55;white-space:pre-wrap;color:var(--ink);font-size:.9rem}
.m-src.cited{border-left-color:#F59E0B;background:#FFFAF0}
.m-src.extra{border-left-style:dashed}
.m-src .meta{font-size:.76rem;color:var(--muted);margin-bottom:.25rem;white-space:normal}
.m-src mark{background:#FDE68A;color:inherit;padding:0 .1em;border-radius:3px}
@media (max-width:900px){.m-cards{grid-template-columns:1fr}.stMain .block-container{margin:0;border-radius:0;padding:3rem 1rem 1rem}
  [data-testid="stBottom"]{padding:0}[data-testid="stBottom"]>div{border-radius:0}.stMain>div:not(.block-container){margin-right:0}
  .m-hello,.m-h1{font-size:1.7rem}}
"""
st.markdown("<style>" + CSS + mascot.CSS + "</style>", unsafe_allow_html=True)
AVATAR = mascot.avatar_data_uri()


# ---------------------------------------------------------------------------
# Secrets, limits and state
# ---------------------------------------------------------------------------

def read_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def limit(name, default):
    try:
        return int(read_secret(name) or default)
    except ValueError:
        return default


LIMITS = {"chat_rpd": limit("GEMINI_DAILY_REQUESTS", usage_mod.DEFAULT_LIMITS["chat_rpd"]),
          "chat_rpm": limit("GEMINI_REQUESTS_PER_MINUTE", usage_mod.DEFAULT_LIMITS["chat_rpm"]),
          "chat_tpm": limit("GEMINI_TOKENS_PER_MINUTE", usage_mod.DEFAULT_LIMITS["chat_tpm"]),
          "embed_rpd": limit("GEMINI_EMBED_DAILY_REQUESTS", usage_mod.DEFAULT_LIMITS["embed_rpd"]),
          "embed_rpm": limit("GEMINI_EMBED_REQUESTS_PER_MINUTE", usage_mod.DEFAULT_LIMITS["embed_rpm"])}

rag.EMBED_DAILY_LIMIT = LIMITS["embed_rpd"]

password = read_secret("APP_PASSWORD")
if password and not st.session_state.get("unlocked"):
    st.markdown(mascot.render("idle", "This Margin is private.", "hero", uid="lock"), unsafe_allow_html=True)
    entered = st.text_input("Access password", type="password")
    if entered == password:
        st.session_state.unlocked = True
        st.rerun()
    elif entered:
        st.error("That password is incorrect. Ask the app owner for the current one.")
    st.stop()

API_KEY = read_secret("GOOGLE_API_KEY")
ss = st.session_state
ss.setdefault("chats", {})
ss.setdefault("current", None)
ss.setdefault("view", "chat")
ss.setdefault("pip", None)


def new_chat():
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": "New chat", "created": time.time(), "messages": [], "entries": [],
                     "suggestions": [], "trace": None}
    ss.current, ss.view, ss.pip = cid, "chat", None
    return cid


if not ss.current or ss.current not in ss.chats:
    new_chat()
chat = ss.chats[ss.current]


def snap():
    return rag.USAGE.snapshot(LIMITS)


def friendly_error(err):
    msg = str(err)
    if rag.is_rate_limit(err):
        u = snap()
        if u["chat_left"] == 0 or "per day" in msg.lower() or "daily" in msg.lower():
            return (f"Today's Gemini allowance is used up. It refreshes in "
                    f"{usage_mod.human_duration(u['day_reset_s'])} (midnight Pacific time).")
        return (f"Gemini's per-minute limit is full. Try again in about "
                f"{usage_mod.human_duration(max(u['minute_reset_s'], 20))}.")
    if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in the app's secrets."
    if "404" in msg or "not found" in msg.lower():
        return "That Gemini model isn't available for this key. Pick another model in Settings."
    return f"Something went wrong while talking to Gemini: {msg[:220]}"


@st.cache_data(show_spinner=False, max_entries=30)
def detect(name, data):
    d = rag.detect_profile(name, data)
    d.pop("signals", None)
    return d


def short(text, n=46):
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def where(chunk, preposition=False):
    page, sec = chunk.get("page"), chunk.get("section") or ""
    if chunk.get("kind") == "footnote":
        return ("in " if preposition else "") + f"endnote {chunk.get('note')}"
    if sec and not sec.lower().startswith("page") and sec != chunk.get("chapter"):
        return ("in " if preposition else "") + (f"{sec}, page {page}" if page else sec)
    return ("on " if preposition else "") + (f"page {page}" if page else short(chunk.get("title", "the document"), 30))


STOP = set("a an and are as at be by can do does for from has have how i in is it its me my of on or should "
           "that the their this to was what when where which who why will with you your".split())


def highlight(text, query):
    safe = html.escape(text)
    words = {w for w in re.findall(r"[A-Za-z0-9.:]+", query.lower()) if len(w) > 2 and w not in STOP}
    for w in sorted(words, key=len, reverse=True):
        safe = re.sub(rf"(?i)\b({re.escape(w)})", r"<mark>\1</mark>", safe)
    return safe


def chat_kb(c, progress=None):
    if not c["entries"]:
        return None
    return rag.build_kb(c["entries"], API_KEY, ss.get("model", CHAT_MODELS[0]), progress=progress)


# ---------------------------------------------------------------------------
# Sidebar: brand, new chat, navigation, history, usage
# ---------------------------------------------------------------------------

def usage_card(u):
    pct = u["chat_pct_left"]
    tone = "" if pct >= .5 else ("amber" if pct >= .2 else "red")
    left_line = (f"≈ {u['questions_left']} questions left today" if u["chat_left"] else "Today's allowance is used up")
    return (f'<div class="m-usage"><div class="h"><span>Gemini usage today</span>'
            f'<span>{int(pct * 100)}% left</span></div>'
            f'<div class="big">{left_line}</div>'
            f'<div class="m-meter {tone}"><i style="width:{max(pct, .02) * 100:.0f}%"></i></div>'
            f'<div class="s">Answers: {u["chat_used"]} of {u["chat_limit"]:,} requests<br>'
            f'Reading: {u["embed_used"]} of {u["embed_limit"]:,} passages<br>'
            f'Refreshes in <b>{usage_mod.human_duration(u["day_reset_s"])}</b> (midnight Pacific)</div></div>')


def usage_details(u):
    st.markdown("**Today** (resets at midnight Pacific, in "
                f"{usage_mod.human_duration(u['day_reset_s'])})")
    st.progress(min(u["chat_used"] / max(u["chat_limit"], 1), 1.0),
                text=f"Answers and reading: {u['chat_used']} of {u['chat_limit']} requests")
    st.progress(min(u["embed_used"] / max(u["embed_limit"], 1), 1.0),
                text=f"Reading (embeddings): {u['embed_used']} of {u['embed_limit']} passages")
    st.caption("Reading uses one request per passage: about 50 for a 12-page report, 230 for 80 pages "
               "and 600 for 200 pages. Each question uses 1 more.")
    st.caption(f"Tokens today: {u['input_tokens']:,} in · {u['output_tokens']:,} out · "
               f"{u['embed_tokens']:,} embedded (estimated)")
    st.markdown("**This minute** (rolling 60 seconds)")
    st.progress(min(u["minute_requests"] / max(u["minute_rpm"], 1), 1.0),
                text=f"{u['minute_requests']} of {u['minute_rpm']} requests · "
                     f"{u['minute_tokens']:,} of {u['minute_tpm']:,} tokens")
    if u["minute_requests"]:
        st.caption(f"Frees up in {usage_mod.human_duration(u['minute_reset_s'])}.")
    st.caption(f"Each question uses about {u['requests_per_question']} requests, so ≈ {u['questions_left']} questions "
               "are left today. Margin counts its own calls; Google counts per project, so other apps using the "
               "same key aren't included. See your exact limits in "
               "[AI Studio → Usage](https://aistudio.google.com/usage).")


U = snap()
with st.sidebar:
    st.markdown(f'<div class="m-brand">{mascot.icon(uid="brand", size=34)}Margin</div>', unsafe_allow_html=True)
    if st.button("New chat", icon=":material/add:", type="primary", width="stretch"):
        new_chat()
        st.rerun()
    query = st.text_input("Search chats", placeholder="Search chats", label_visibility="collapsed",
                          icon=":material/search:")
    nav = [("chat", "Chat", ":material/chat_bubble:"), ("docs", "Documents", ":material/description:"),
           ("how", "How it works", ":material/account_tree:"), ("test", "Test it", ":material/fact_check:")]
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    for key, label, ic in nav:
        n = f" ({len(chat['entries'])})" if key == "docs" and chat["entries"] else ""
        if st.button(label + n, icon=ic, type="tertiary", key=f"nav_{key}", width="stretch"):
            ss.view = key
            st.rerun()

    listed = [(cid, c) for cid, c in ss.chats.items() if c["messages"]
              and (not query or query.lower() in c["title"].lower())]
    listed.sort(key=lambda x: -x[1]["created"])
    today = dt.date.today()
    groups = {}
    for cid, c in listed:
        d = dt.date.fromtimestamp(c["created"])
        g = "Today" if d == today else ("Yesterday" if (today - d).days == 1 else "Earlier")
        groups.setdefault(g, []).append((cid, c))
    if listed:
        st.divider()
    for g in ("Today", "Yesterday", "Earlier"):
        if g in groups:
            st.markdown(f'<div class="m-group">{g}</div>', unsafe_allow_html=True)
            for cid, c in groups[g]:
                if st.button(short(c["title"], 34), key=f"chat_{cid}", type="tertiary", width="stretch"):
                    ss.current, ss.view, ss.pip = cid, "chat", None
                    st.rerun()
    if not listed and query:
        st.caption("No chats match.")

    st.markdown(usage_card(U), unsafe_allow_html=True)
    with st.popover("Usage details", icon=":material/data_usage:", width="stretch"):
        usage_details(U)
    st.caption("Chats are kept for this browser session.")

# views without the message bar get a full-height panel with all corners rounded
if ss.view != "chat":
    st.markdown("<style>.stMain .block-container{min-height:calc(100vh - 28px);border-radius:28px;margin-bottom:14px}</style>",
                unsafe_allow_html=True)

# highlight the active nav item and chat
st.markdown(f"<style>.st-key-nav_{ss.view} button,.st-key-chat_{ss.current} button"
            "{background:#F1ECFD!important;color:#5B21B6!important;font-weight:600!important}</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top bar: chat title + documents, usage pill, export, settings
# ---------------------------------------------------------------------------

top_l, top_r = st.columns([2, 3], vertical_alignment="center")
with top_l:
    docs_html = "".join(f'<span class="m-badge {e["profile"]}">{TYPE_SHORT[e["profile"]]}</span>'
                        for e in chat["entries"])
    title = chat["title"] if chat["messages"] else ""
    st.markdown(f'<div class="m-top"><span class="ttl">{html.escape(short(title, 48))}</span>{docs_html}</div>',
                unsafe_allow_html=True)
with top_r:
    _, c1, c2, c3 = st.columns([1.2, 1.1, 1, 1.1], gap="small")
    dot = "🟢" if U["chat_pct_left"] >= .5 else ("🟠" if U["chat_pct_left"] >= .2 else "🔴")
    with c1.popover(f"{dot} {U['questions_left']} left", width="stretch", help="Questions left today"):
        usage_details(U)
    export = "\n\n".join((f"**You:** {m['content']}" if m["role"] == "user" else f"**Margin:** {m['content']}")
                         for m in chat["messages"] if m.get("content"))
    c2.download_button("Export", data=f"# {chat['title']}\n\n{export}", file_name="margin-chat.md",
                       icon=":material/download:", width="stretch", disabled=not chat["messages"])
    with c3.popover("Settings", icon=":material/tune:", width="stretch"):
        st.markdown("**Settings**")
        st.selectbox("Model", CHAT_MODELS, key="model",
                      help="Flash-lite is faster and uses less of your allowance. Flash writes fuller answers.")
        st.slider("Answer style", 0.0, 1.0, 0.1, 0.1, key="temperature",
                  help="Temperature. Low keeps answers factual and consistent, which suits document Q&A.")
        st.caption("Low = precise and consistent · High = more varied wording")
        st.slider("Passages per answer", 2, 10, 6, key="k",
                  help="Top-k: how many passages Pip reads before answering.")
        st.caption("More passages = fuller context but slower answers")
        with st.expander("Pipeline steps"):
            st.toggle("Hybrid search (meaning + exact words)", True, key="use_hybrid")
            st.toggle("Section-aware routing", True, key="use_routing")
            st.toggle("Rerank with Gemini", True, key="use_rerank", help="One extra request per question.")
            st.toggle("Add neighbouring passages and footnotes", True, key="use_expand")
            st.toggle("Understand follow-up questions", True, key="use_rewrite")
        st.caption("Files are sent to Google Gemini. Don't upload confidential or client data.")

model = ss.get("model", CHAT_MODELS[0])
temperature = ss.get("temperature", 0.1)
settings = dict(k=ss.get("k", 6), use_routing=ss.get("use_routing", True), use_hybrid=ss.get("use_hybrid", True),
                use_rerank=ss.get("use_rerank", True), use_expand=ss.get("use_expand", True))
use_rewrite = ss.get("use_rewrite", True)

if not API_KEY:
    st.markdown(mascot.render("confused", "I need a Gemini API key first.", "hero", uid="nokey"), unsafe_allow_html=True)
    st.error('Add GOOGLE_API_KEY = "your-key" under Settings → Secrets in Streamlit Cloud.')
    st.stop()

kb = None
if chat["entries"]:
    try:
        kb = chat_kb(chat)
    except Exception as err:
        st.error(friendly_error(err))

pip_slot = st.empty()


def pip(mood, message, autohide=False):
    ss.pip = (mood, message, autohide)
    pip_slot.markdown(mascot.render(mood, message, "dock", autohide, uid="dock"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(kb, question, history, opts, stream_to=None, narrate=False):
    before = rag.USAGE.chat_requests_so_far()
    llm, llm0, times = rag.make_llm(API_KEY, model, temperature), rag.make_llm(API_KEY, model, 0.0), {}
    if narrate:
        pip("thinking", random.choice(["Hmm, let me look…", f"Searching {len(kb['chunks'])} passages…",
                                       "Flipping through the pages…"]))
    t = time.time()
    query = rag.rewrite_query(llm0, question, history) if (use_rewrite and history) else question
    times["rewrite"] = time.time() - t
    t = time.time()
    results, trace = rag.search(kb, query, API_KEY, llm=llm0, **opts)
    times["retrieve"] = time.time() - t
    if narrate:
        pip("talking", f"Looking only in {', '.join(short(r, 30) for r in trace['routes'])}. Writing the answer…"
            if trace["routes"] else "Found some good passages. Writing the answer…")
    msgs = rag.build_messages(kb, question, results, history)
    t = time.time()
    if stream_to is not None:
        def gen():
            for chunk in llm.stream(msgs):
                text = rag.get_text(chunk)
                if text:
                    yield text
        answer = stream_to.write_stream(gen())
    else:
        answer = rag.get_text(rag.call_with_retry(lambda: llm.invoke(msgs)))
    times["generate"] = time.time() - t
    rag.USAGE.record_question(rag.USAGE.chat_requests_so_far() - before)
    trace.update(question=question, times=times, results=results, answer=answer,
                 prompt="\n\n".join(f"[{m.type}]\n{m.content}" for m in msgs))
    return answer, results, trace


def snippet(text, query, size=460):
    """A readable excerpt centred on the first matching word, with PDF line breaks joined up."""
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    if len(flat) <= size:
        return flat
    words = [w for w in re.findall(r"[A-Za-z0-9.:]+", query.lower()) if len(w) > 2 and w not in STOP]
    hits = [m.start() for w in words for m in [re.search(rf"(?i)\b{re.escape(w)}", flat)] if m]
    start = max(0, min(hits) - size // 3) if hits else 0
    start = flat.rfind(" ", 0, start) + 1 if start else 0
    excerpt = flat[start:start + size]
    excerpt = excerpt[: excerpt.rfind(" ")] if start + size < len(flat) else excerpt
    return ("…" if start else "") + excerpt + ("…" if start + size < len(flat) else "")


def render_sources(kb, results, query, answer):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
    numbered = [(i, r) for i, r in enumerate(results, 1) if r["id"] < len(kb["chunks"])]
    used = [(i, r) for i, r in numbered if i in cited]
    rest = [(i, r) for i, r in numbered if i not in cited]
    label = f"Sources · {len(used)} cited" + (f", {len(rest)} more read for context" if rest else "")
    with st.expander(label):
        for i, r in used or numbered[:2]:
            c = kb["chunks"][r["id"]]
            st.markdown(f'<div class="m-src cited"><div class="meta"><b>[{i}]</b> {html.escape(c["header"])} · '
                        f'{KIND[c["kind"]]}</div>{highlight(snippet(c["text"], query), query)}</div>',
                        unsafe_allow_html=True)
        if rest and used:
            items = "".join(f"<li><b>[{i}]</b> {html.escape(where(kb['chunks'][r['id']]))}"
                            f"{' · ' + html.escape(r['reason']) if r['reason'] != 'retrieved' else ''}</li>"
                            for i, r in rest)
            st.markdown(f'<div class="meta" style="color:var(--muted);font-size:.8rem">Also read for context:'
                        f'<ul style="margin:.3rem 0 0 1rem">{items}</ul></div>', unsafe_allow_html=True)


def doc_card(e):
    chips = "".join(f'<span class="m-chip">{html.escape(r)}</span>' for r in e.get("reasons", []))
    return (f'<div class="m-doc"><div class="t">{html.escape(short(e["title"], 70))}</div>'
            f'<span class="m-badge {e["profile"]}">{TYPE_SHORT[e["profile"]]}</span>'
            f'<span class="n">{html.escape(e["name"])}</span><div style="margin-top:.35rem">{chips}</div></div>')


def starters(c, kb):
    known = [e["known"] for e in c["entries"] if e.get("known")]
    qs = [q for k in known for q in rag.USE_CASES[k]["starters"]]
    return (qs or c.get("suggestions") or ["Summarise the main points of this document."])[:3]


# ---------------------------------------------------------------------------
# View: Chat
# ---------------------------------------------------------------------------

def view_chat():
    center = st.container(key="center")
    with center:
        if not chat["messages"]:
            if kb is None:
                st.markdown(mascot.render("greet", "Hello!", "hero", uid="hero"), unsafe_allow_html=True)
                st.markdown('<div class="m-hello">Hi, I\'m Pip</div>'
                            '<div class="m-h1">What would you like to read today?</div>'
                            '<p class="m-sub">Attach a document and ask anything. I work out what kind of document it '
                            'is and read it the right way, then show you the exact passage behind every answer.</p>',
                            unsafe_allow_html=True)
                cards = [("narrative", "Plain reports", "Annual reports, memos, articles. I quote the exact sentence."),
                         ("structured", "Long, structured reports", "Chapters, regions, boxes. Answers stay in the right section."),
                         ("visual", "Charts, tables & footnotes", "Designed reports. I read each page image so no number hides.")]
                st.markdown('<div class="m-cards">' + "".join(
                    f'<div class="m-card"><span class="ic">{icon(k)}</span><h4>{t}</h4><p>{d}</p></div>'
                    for k, t, d in cards) + "</div>", unsafe_allow_html=True)
                st.markdown('<p class="m-hint">Attach a PDF, Word, Markdown or text file with the 📎 in the message bar '
                            'below.</p>', unsafe_allow_html=True)
            else:
                st.markdown(mascot.render("happy", "Ready!", "hero", uid="hero"), unsafe_allow_html=True)
                st.markdown(f'<div class="m-hello">All read</div><div class="m-h1">'
                            f'{html.escape(short(kb["title"], 60))}</div>'
                            f'<p class="m-sub">{len(kb["chunks"])} passages. Ask anything, or start with one of these.</p>',
                            unsafe_allow_html=True)
                with st.container(key="sugg"):
                    cols = st.columns(3)
                    for i, (col, q) in enumerate(zip(cols, starters(chat, kb))):
                        if col.button(q, key=f"s_{i}", width="stretch"):
                            ss.pending = q
                            st.rerun()
        else:
            last_assistant = max((i for i, m in enumerate(chat["messages"]) if m["role"] == "assistant"), default=-1)
            asked = any(m["role"] == "user" and m.get("content") for m in chat["messages"])
            for i, m in enumerate(chat["messages"]):
                with st.chat_message(m["role"], avatar=AVATAR if m["role"] == "assistant" else None):
                    if m.get("files"):
                        st.markdown(" ".join(f'<span class="m-chip">📎 {html.escape(f)}</span>' for f in m["files"]),
                                    unsafe_allow_html=True)
                    if m.get("content"):
                        st.markdown(m["content"])
                    if m.get("docs"):
                        st.markdown("".join(doc_card(e) for e in m["docs"]), unsafe_allow_html=True)
                    if m.get("results") and kb:
                        render_sources(kb, m["results"], m.get("query", ""), m["content"])
                    if m.get("docs") and i == last_assistant and not asked and kb:
                        with st.container(key=f"sugg_after_{i}"):
                            for j, q in enumerate(starters(chat, kb)):
                                if st.button(q, key=f"a_{i}_{j}", icon=":material/arrow_outward:"):
                                    ss.pending = q
                                    st.rerun()

    if ss.pip and chat["messages"]:
        pip(*ss.pip)
    elif chat["messages"]:
        pip("idle", f"Ask me anything about {short(kb['title'], 40)}." if kb else "Attach a document to begin.", True)

    if U["chat_left"] == 0:
        st.markdown(f'<div class="m-banner">Today\'s Gemini allowance is used up. It refreshes in '
                    f'<b>{usage_mod.human_duration(U["day_reset_s"])}</b> (midnight Pacific).</div>',
                    unsafe_allow_html=True)
    elif U["minute_requests"] >= U["minute_rpm"]:
        st.markdown(f'<div class="m-banner">Gemini\'s per-minute limit is full. You can ask again in about '
                    f'<b>{usage_mod.human_duration(U["minute_reset_s"])}</b>.</div>', unsafe_allow_html=True)
    elif U["questions_left"] <= 10:
        st.markdown(f'<div class="m-banner">Only about {U["questions_left"]} questions left today. '
                    f'Refreshes in {usage_mod.human_duration(U["day_reset_s"])}.</div>', unsafe_allow_html=True)

    placeholder = "Ask a question, or attach a document" if kb else "Attach a document with 📎, then ask anything"
    sent = st.chat_input(placeholder, accept_file="multiple", file_type=["pdf", "docx", "md", "txt"],
                         max_upload_size=100)
    pending = ss.pop("pending", None)
    if not sent and not pending:
        return
    text = (sent.text if sent else pending) or ""
    files = list(sent.files) if sent else []
    handle(text.strip(), files)


def handle(text, files):
    global kb
    names = [f.name for f in files]
    earlier = [m for m in chat["messages"] if m.get("content") and not m.get("docs")]  # before this question
    chat["messages"].append({"role": "user", "content": text, "files": names})
    with st.container(key="center_live"):
        with st.chat_message("user"):
            if names:
                st.markdown(" ".join(f'<span class="m-chip">📎 {html.escape(n)}</span>' for n in names),
                            unsafe_allow_html=True)
            if text:
                st.markdown(text)

        if files:
            with st.chat_message("assistant", avatar=AVATAR):
                existing = {hashlib.md5(e["data"]).hexdigest() for e in chat["entries"]}
                new = []
                for f in files:
                    data = f.getvalue()
                    if hashlib.md5(data).hexdigest() in existing:
                        continue
                    d = detect(f.name, data)
                    new.append({"name": f.name, "data": data, "profile": d["profile"], "title": d["title"],
                                "known": d.get("known"), "reasons": d["reasons"]})
                if not new:
                    st.markdown("I've already read that document in this chat.")
                    chat["messages"].append({"role": "assistant", "content": "I've already read that document in this chat."})
                else:
                    if len(new) == 1:
                        intro = (f"This looks like **{TYPE_PLAIN[new[0]['profile']]}**. {TYPE_PLAN[new[0]['profile']]}")
                    else:
                        intro = "Here's what I found. I'll read each one with its own strategy."
                    st.markdown(intro)
                    st.markdown("".join(doc_card(e) for e in new), unsafe_allow_html=True)
                    pip("reading", f"Reading {short(new[0]['title'], 34)}…")
                    status = st.status(f"Reading {len(new)} document{'s' if len(new) > 1 else ''}…", expanded=True)
                    bar = status.progress(0.0)
                    line = status.empty()
                    moods = {"parse": "reading", "vision": "reading", "embed": "thinking", "wait": "confused"}

                    def progress(stage, message, fraction, file=None):
                        line.caption(message)
                        bar.progress(min(max(fraction, 0.0), 1.0))
                        pip(moods.get(stage, "reading"), message)

                    chat["entries"] = chat["entries"] + new
                    try:
                        kb = chat_kb(chat, progress=progress)
                    except Exception as err:
                        chat["entries"] = [e for e in chat["entries"] if e not in new]
                        status.update(label="Couldn't read the document", state="error")
                        msg = str(err) if isinstance(err, ValueError) else friendly_error(err)
                        st.error(msg)
                        pip("confused", msg)
                        chat["messages"].append({"role": "assistant", "content": msg})
                        st.rerun()
                    n_vis = sum(1 for c in kb["chunks"] if c["kind"] == "visual")
                    status.update(label=f"Read {len(new)} document{'s' if len(new) > 1 else ''} · "
                                        f"{len(kb['chunks'])} passages", state="complete", expanded=False)
                    if not any(e.get("known") for e in chat["entries"]) and not chat["suggestions"]:
                        chat["suggestions"] = rag.suggest_questions(kb, rag.make_llm(API_KEY, model, 0.4))
                    if chat["title"] == "New chat":
                        chat["title"] = new[0]["title"]
                    chat["messages"].append({"role": "assistant", "content": intro, "docs": new})
                    extra = f", including {n_vis} pages of charts and tables" if n_vis else ""
                    pip("happy", f"All read: {len(kb['chunks'])} passages{extra}."
                        + ("" if text else " Ask me anything!"), True)

        if text:
            if chat["title"] == "New chat":
                chat["title"] = text
            with st.chat_message("assistant", avatar=AVATAR):
                if kb is None:
                    reply = ("I answer from your documents, so I need one first. Attach a PDF, Word, Markdown or "
                             "text file with the 📎 in the message bar.")
                    st.markdown(reply)
                    chat["messages"].append({"role": "assistant", "content": reply})
                    pip("idle", "Attach a document and I'll read it for you.")
                elif snap()["chat_left"] == 0:
                    reply = friendly_error(Exception("429 daily"))
                    st.markdown(reply)
                    chat["messages"].append({"role": "assistant", "content": reply})
                    pip("confused", reply)
                else:
                    try:
                        answer, results, trace = run_pipeline(kb, text, earlier, settings, stream_to=st, narrate=True)
                    except Exception as err:
                        msg = friendly_error(err)
                        st.error(msg)
                        pip("confused", msg)
                        st.stop()
                    chat["messages"].append({"role": "assistant", "content": answer, "results": results,
                                             "query": trace["query"]})
                    chat["trace"] = trace
                    if rag.NOT_FOUND.lower()[:20] in answer.lower():
                        pip("confused", "I couldn't find that in your documents, so I won't guess. Try asking another way?")
                    else:
                        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
                        top = results[cited[0] - 1] if cited and cited[0] <= len(results) else (results[0] if results else None)
                        spot = f" {where(kb['chunks'][top['id']], True)}" if top else ""
                        pip("happy", f"Found it{spot}. The exact passages are under my answer.", True)
    st.rerun()


# ---------------------------------------------------------------------------
# View: Documents
# ---------------------------------------------------------------------------

def view_docs():
    with st.container(key="center"):
        st.markdown('<div class="m-h1" style="text-align:left;font-size:1.7rem">Documents in this chat</div>',
                    unsafe_allow_html=True)
        if not chat["entries"]:
            st.markdown('<p style="color:var(--muted)">No documents yet. Attach one with the 📎 in the Chat message bar.</p>',
                        unsafe_allow_html=True)
            if st.button("Go to chat", type="primary"):
                ss.view = "chat"
                st.rerun()
            return
        st.caption("Pip picked a reading strategy for each file. If one looks wrong, change it and Pip re-reads "
                   "just that file.")
        for i, e in enumerate(list(chat["entries"])):
            st.markdown(doc_card(e), unsafe_allow_html=True)
            a, b, c = st.columns([2.2, 1, 1], vertical_alignment="bottom")
            choice = a.selectbox("Reading strategy", list(rag.PROFILES), index=list(rag.PROFILES).index(e["profile"]),
                                 format_func=lambda p: rag.PROFILES[p]["label"], key=f"type_{i}_{e['name']}")
            if b.button("Re-read", key=f"reread_{i}", disabled=choice == e["profile"], width="stretch"):
                chat["entries"][i] = {**e, "profile": choice}
                with st.spinner(f"Re-reading {e['name']} as {rag.PROFILES[choice]['label'].lower()}…"):
                    try:
                        chat_kb(chat)
                    except Exception as err:
                        chat["entries"][i] = e
                        st.error(friendly_error(err))
                        return
                ss.pip = ("happy", f"Re-read {short(e['name'], 30)} as {TYPE_SHORT[choice].lower()}.", True)
                st.rerun()
            if c.button("Remove", key=f"rm_{i}", width="stretch", icon=":material/delete:"):
                chat["entries"].pop(i)
                st.rerun()


# ---------------------------------------------------------------------------
# View: How it works
# ---------------------------------------------------------------------------

def need_docs():
    st.markdown(mascot.render("idle", "Attach a document in Chat first.", "hero", uid="need"), unsafe_allow_html=True)
    if st.button("Go to chat", type="primary"):
        ss.view = "chat"
        st.rerun()


def view_how():
    with st.container(key="center"):
        st.markdown('<div class="m-h1" style="text-align:left;font-size:1.7rem">How Pip answered</div>',
                    unsafe_allow_html=True)
        if kb is None:
            need_docs()
            return
        for f in kb["files"]:
            p = rag.PROFILES[f["profile"]]
            steps = "".join(f"<p><b style='color:var(--ink)'>{html.escape(n)}.</b> {html.escape(t)}</p>"
                            for n, t in p["strategy"])
            st.markdown(f'<div class="m-card" style="margin:.5rem 0"><h4><span class="m-badge {f["profile"]}">'
                        f'{TYPE_SHORT[f["profile"]]}</span>{html.escape(short(f["title"], 60))}</h4>{steps}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="m-step">1. Parse and chunk</div>', unsafe_allow_html=True)
        kinds = pd.Series([c["kind"] for c in kb["chunks"]]).value_counts().to_dict()
        st.write(f"**{len(kb['chunks'])}** passages, averaging {int(np.mean([len(c['text']) for c in kb['chunks']]))} "
                 "characters: " + ", ".join(f"{v} {KIND[k2]}" for k2, v in kinds.items()) + ".")
        structured = [c for c in kb["chunks"] if c.get("profile") == "structured"]
        if structured:
            rows = {}
            for c in structured:
                key = (c.get("chapter", ""), c.get("section", ""))
                r = rows.setdefault(key, {"Chapter": key[0], "Section": key[1], "Pages": [], "Passages": 0})
                r["Pages"].append(c.get("page") or 0)
                r["Passages"] += 1
            st.caption("Document map detected while parsing")
            st.dataframe(pd.DataFrame([{**r, "Pages": f"{min(r['Pages'])}–{max(r['Pages'])}"} for r in rows.values()]),
                         hide_index=True, width="stretch")
        kind_filter = st.selectbox("Browse passages of type", ["all"] + list(kinds), format_func=lambda x: KIND.get(x, x))
        pool = [c for c in kb["chunks"] if kind_filter == "all" or c["kind"] == kind_filter]
        ix = st.number_input(f"Passage (1–{len(pool)})", 1, len(pool), 1) - 1
        chosen = pool[ix]
        st.markdown(f'<div class="m-src"><div class="meta">{html.escape(chosen["header"])} · {KIND[chosen["kind"]]}</div>'
                    f'{html.escape(chosen["text"])}</div>', unsafe_allow_html=True)

        st.markdown('<div class="m-step">2. Embed and store</div>', unsafe_allow_html=True)
        st.write(f"Each passage and its header became **{kb['vectors'].shape[1]}** numbers with `{kb['embed_model']}`, "
                 "kept in an in-memory vector store beside a BM25 keyword index. This passage starts with:")
        st.code(", ".join(f"{x:.4f}" for x in kb["vectors"][chosen["id"]][:8]) + ", …")

        trace = chat.get("trace")
        st.markdown('<div class="m-step">3. Retrieve, rerank and assemble context</div>', unsafe_allow_html=True)
        if not trace:
            st.info("Ask a question in Chat to see how it was answered.")
            return
        st.write(f"Question: **{trace['question']}**")
        if trace["query"] != trace["question"]:
            st.write(f"Rewritten for search: **{trace['query']}**")
        if trace.get("routes"):
            st.write("Search limited to: **" + "**, **".join(trace["routes"]) + "**")

        def label(cid):
            return short(where(kb["chunks"][cid]), 40) if cid < len(kb["chunks"]) else "—"

        a, b, c = st.columns(3)
        a.caption("Meaning search")
        a.write("\n".join(f"- {label(i)}" for i in trace["dense"][:5]) or "—")
        b.caption("Keyword search")
        b.write("\n".join(f"- {label(i)}" for i in trace["sparse"][:5]) or "off")
        c.caption("After fusion" + (" + reranking" if trace["reranked"] else ""))
        c.write("\n".join(f"- {label(r['id'])}" for r in trace["results"] if r["reason"] == "retrieved"))
        extras = [r for r in trace["results"] if r["reason"] != "retrieved"]
        if extras:
            st.write("Added for context: " + "; ".join(f"{label(r['id'])} ({r['reason']})" for r in extras))
        st.caption(f"Context sent to Gemini: {trace['context_chars']:,} characters · retrieval "
                   f"{trace['times']['retrieve']:.1f}s · answer {trace['times']['generate']:.1f}s")
        st.markdown('<div class="m-step">4. Generate</div>', unsafe_allow_html=True)
        with st.expander("See the exact prompt sent to Gemini"):
            st.code(trace["prompt"], language=None)


# ---------------------------------------------------------------------------
# View: Test it
# ---------------------------------------------------------------------------

COLS = ["Question", "Must contain", "Expected section", "Should say it doesn't know"]
BASIC = dict(k=4, use_routing=False, use_hybrid=False, use_rerank=False, use_expand=False)


def view_test():
    with st.container(key="center"):
        st.markdown('<div class="m-h1" style="text-align:left;font-size:1.7rem">Test the answers</div>',
                    unsafe_allow_html=True)
        if kb is None:
            need_docs()
            return
        known = [e["known"] for e in chat["entries"] if e.get("known")]
        defaults = [t for k in known for t in rag.USE_CASES[k]["tests"]]
        if defaults:
            st.caption("Pip recognised the hands-on task documents and loaded their sample questions and expected answers.")
        else:
            defaults = [(q, "", "", False) for q in chat.get("suggestions", [])] + [
                ("What is the capital of France?", "", "", True)]
            st.caption("Add the facts each answer must contain to turn these into real checks.")
        st.caption("Must contain: separate required items with ; and alternatives with |  (e.g. 346; 45; 50). "
                   "Expected section is optional and checks retrieval found the right part of the document.")
        cases = st.data_editor(pd.DataFrame(defaults, columns=COLS), num_rows="dynamic", width="stretch",
                               hide_index=True, key=f"tests_{kb['key']}")
        mode = st.radio("Pipeline to test", ["Advanced (current settings)", "Compare with basic RAG"], horizontal=True,
                        help="Basic = meaning search only, top 4, no routing, reranking or context expansion.")
        todo = [r for _, r in cases.iterrows() if str(r["Question"] or "").strip()]
        variants = [("Advanced", settings)] + ([("Basic", BASIC)] if mode.startswith("Compare") else [])
        cost = sum(len(todo) * (3 if v[1].get("use_rerank") else 2) for v in variants)
        share = cost / max(U["chat_limit"], 1)
        st.caption(f"This run uses about **{cost} requests** ({share:.0%} of today's allowance; "
                   f"{U['chat_left']} left).")
        if st.button("Run tests", type="primary", disabled=cost > U["chat_left"]):
            rows, total, done = [], len(todo) * len(variants), 0
            bar_t = st.progress(0.0, text="Running tests…")
            pip("thinking", f"Running {total} tests. I'll mark my own homework.")
            for vname, opts in variants:
                for r in todo:
                    q = str(r["Question"]).strip()
                    try:
                        t = time.time()
                        answer, results, _ = run_pipeline(kb, q, [], opts)
                        ok, why = rag.check_answer(answer, str(r["Must contain"] or ""),
                                                   bool(r["Should say it doesn't know"]))
                        ret = rag.check_retrieval(kb, results, str(r["Expected section"] or ""))
                        rows.append({"Pipeline": vname, "Result": "✅ Pass" if ok else "❌ Fail", "Question": q,
                                     "Why": why, "Right section": "—" if ret is None else ("✅" if ret else "❌"),
                                     "Top source": where(kb["chunks"][results[0]["id"]]) if results else "",
                                     "Seconds": round(time.time() - t, 1), "Answer": answer})
                    except Exception as e:
                        rows.append({"Pipeline": vname, "Result": "⚠️ Error", "Question": q, "Why": friendly_error(e)})
                    done += 1
                    bar_t.progress(done / total, text=f"Ran {done} of {total}")
            bar_t.empty()
            df = pd.DataFrame(rows)
            scores = {}
            for col, (vname, _) in zip(st.columns(len(variants)), variants):
                part = df[df["Pipeline"] == vname]
                scores[vname] = (int((part["Result"] == "✅ Pass").sum()), len(part))
                col.metric(f"{vname} pipeline", f"{scores[vname][0]} of {scores[vname][1]} passed")
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption("A ❌ in Right section is a retrieval problem. A ✅ section with a wrong answer is a generation problem.")
            adv = scores["Advanced"]
            if adv[0] == adv[1]:
                pip("happy", f"Perfect score: {adv[0]} of {adv[1]}!", True)
            elif "Basic" in scores and adv[0] > scores["Basic"][0]:
                pip("happy", f"The advanced pipeline beat basic RAG, {adv[0]} to {scores['Basic'][0]}.", True)
            else:
                pip("confused", f"{adv[0]} of {adv[1]} passed. The Why column shows what to fix.")


{"chat": view_chat, "docs": view_docs, "how": view_how, "test": view_test}[ss.view]()
