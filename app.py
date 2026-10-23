"""
Margin — ask your documents.

Upload any PDF, Word, Markdown or text file. Margin detects what kind of document it is
(plain narrative, long structured report, or visual with charts/tables/footnotes) and picks
the matching RAG strategy. Pip, the mascot, narrates what's happening.

Pipeline (rag_engine.py):
Document -> Parsing -> Chunking -> Embedding -> Retrieval -> Reranking -> Context -> LLM -> Evaluation
"""

import hashlib
import html
import os
import random
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

import mascot
import rag_engine as rag

CHAT_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
TYPE_SHORT = {"narrative": "Narrative", "structured": "Structured", "visual": "Visual"}
TYPE_LINE = {
    "narrative": "This reads like a straight narrative report. I'll go page by page and keep every fact next to its sentence.",
    "structured": "A long, structured one! I'll map every chapter and section first, so I never mix them up.",
    "visual": "Lots of charts, tables and footnotes here. I'll look at every page image so no number can hide from me.",
}
KIND = {"text": "text", "table": "table", "visual": "chart/table from page image", "footnote": "endnote"}

st.set_page_config(page_title="Margin · Ask your documents", page_icon="🖍️", layout="centered",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Visual design: deep ink background with a soft violet-to-sky aurora
# ---------------------------------------------------------------------------

st.markdown("<style>" + """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root{--ink:#0E1222;--panel:rgba(255,255,255,.045);--line:rgba(255,255,255,.09);--text:#E7E9F3;--muted:#9AA3B8;
  --violet:#7C5CFF;--sky:#38BDF8;--grad:linear-gradient(135deg,#7C5CFF 0%,#4F8BFF 55%,#38BDF8 100%)}
.stApp{background:
  radial-gradient(900px 480px at 8% -8%,rgba(124,92,255,.30),transparent 60%),
  radial-gradient(760px 420px at 100% 0%,rgba(56,189,248,.20),transparent 60%),
  radial-gradient(700px 500px at 50% 115%,rgba(244,114,182,.10),transparent 60%),var(--ink);
  background-attachment:fixed;color:var(--text)}
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button p,.stApp span{
  font-family:'Plus Jakarta Sans',system-ui,sans-serif}
[data-testid="stIconMaterial"],.material-symbols-rounded{font-family:'Material Symbols Rounded'!important}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.4rem;padding-bottom:9rem;max-width:860px}
[data-testid="stSidebar"]{background:rgba(17,21,40,.82);border-right:1px solid var(--line)}
.m-word{font-weight:700;font-size:2.5rem;letter-spacing:-.03em;line-height:1;margin:0;
  background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;display:inline-block}
.m-tag{color:var(--muted);margin:.6rem 0 1rem 0;font-size:1rem;max-width:56ch}
.m-hero-title{font-size:1.9rem;font-weight:700;letter-spacing:-.025em;line-height:1.2;margin:.2rem 0 .5rem 0;color:var(--text)}
.m-card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;margin:.6rem 0}
.m-card h4{margin:0 0 .25rem 0;font-size:1rem;font-weight:600;color:var(--text)}
.m-card p{margin:.15rem 0;color:var(--muted);font-size:.88rem}
.m-badge{display:inline-block;font-size:.75rem;font-weight:600;padding:.18rem .65rem;border-radius:999px;color:#0E1222;margin-right:.4rem}
.m-badge.narrative{background:linear-gradient(135deg,#C4B5FD,#A78BFA)}
.m-badge.structured{background:linear-gradient(135deg,#7DD3FC,#5EEAD4)}
.m-badge.visual{background:linear-gradient(135deg,#FDBA74,#F9A8D4)}
.m-chip{display:inline-block;font-size:.78rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;
  padding:.12rem .6rem;margin:.15rem .3rem .15rem 0;background:rgba(255,255,255,.02)}
.m-bar{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .4rem 0}
.m-step{font-weight:600;font-size:1.05rem;margin:1.2rem 0 .25rem 0;color:var(--text)}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);background:var(--panel);
  color:var(--text);transition:border-color .2s,transform .1s}
.stButton>button:hover{border-color:rgba(124,92,255,.6);color:#fff}
.stButton>button:active{transform:scale(.98)}
.stButton>button[kind="primary"]{background:var(--grad);border:0;color:#fff;font-weight:600;
  box-shadow:0 8px 24px rgba(79,139,255,.28)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08)}
.stButton>button:focus-visible{outline:2px solid #A78BFA;outline-offset:2px}
[data-testid="stFileUploaderDropzone"]{background:rgba(124,92,255,.06);border:1.5px dashed rgba(167,139,250,.45);border-radius:18px;
  padding:1.6rem 1.2rem;min-height:110px;transition:border-color .2s,background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:rgba(167,139,250,.85);background:rgba(124,92,255,.10)}
.st-key-starters .stButton>button{justify-content:flex-start;text-align:left;padding:.7rem 1rem}
.stButton>button p{white-space:normal;text-align:inherit}
[data-testid^="stChatMessageAvatar"]{background:var(--grad)!important;color:#fff!important;border-radius:10px}
[data-testid="stChatMessageAvatarUser"]{background:rgba(255,255,255,.10)!important}
.stApp p.m-label{color:var(--muted);font-size:.8rem!important;margin:.15rem 0 .1rem .15rem;line-height:1.3}
[data-testid="stChatMessage"]{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:.9rem 1rem}
[data-testid="stChatInput"]{border-radius:16px}
[data-testid="stChatInput"] textarea{font-family:'Plus Jakarta Sans',system-ui,sans-serif}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line);border-radius:14px}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{padding:.4rem .9rem;border-radius:10px 10px 0 0}
.stTabs [data-baseweb="tab-highlight"]{background:var(--grad);height:3px;border-radius:3px}
[data-testid="stProgress"] div[role="progressbar"] div div{background:var(--grad)!important}
.m-src{border-left:3px solid #6D7BFF;padding:.6rem .9rem;margin:.5rem 0 .8rem 0;background:rgba(109,123,255,.07);
  border-radius:0 12px 12px 0;line-height:1.55;white-space:pre-wrap;color:var(--text)}
.m-src.cited{border-left-color:#FDBA74;background:rgba(253,186,116,.08)}
.m-src.extra{border-left-style:dashed}
.m-src .meta{font-size:.8rem;color:var(--muted);margin-bottom:.3rem;white-space:normal}
.m-src mark{background:rgba(253,186,116,.35);color:inherit;padding:0 .1em;border-radius:3px}
""" + mascot.CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Secrets and helpers
# ---------------------------------------------------------------------------

def read_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


password = read_secret("APP_PASSWORD")
if password and not st.session_state.get("unlocked"):
    st.markdown('<div class="m-word">Margin</div>', unsafe_allow_html=True)
    st.markdown(mascot.render("idle", "This Margin is private. Enter the access password to continue.", "hero"),
                unsafe_allow_html=True)
    entered = st.text_input("Access password", type="password")
    if entered == password:
        st.session_state.unlocked = True
        st.rerun()
    elif entered:
        st.error("That password is incorrect. Ask the app owner for the current one.")
    st.stop()

API_KEY = read_secret("GOOGLE_API_KEY")
ss = st.session_state
for k, v in {"messages": [], "last_trace": None, "entries": None, "kb_key": None, "editing": False,
             "uploader_n": 0, "pip": None}.items():
    ss.setdefault(k, v)


def friendly_error(err):
    msg = str(err)
    if rag.is_rate_limit(err):
        return "Gemini's free-tier limit was reached. Wait a minute and try again."
    if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in the app's secrets."
    if "404" in msg or "not found" in msg.lower():
        return "That Gemini model isn't available for this key. Pick another model in Settings."
    return f"Something went wrong while talking to Gemini: {msg[:240]}"


STOP = set("a an and are as at be by can do does for from has have how i in is it its me my of on or should "
           "that the their this to was what when where which who why will with you your".split())


def highlight(text, query):
    safe = html.escape(text)
    words = {w for w in re.findall(r"[A-Za-z0-9.:]+", query.lower()) if len(w) > 2 and w not in STOP}
    for w in sorted(words, key=len, reverse=True):
        safe = re.sub(rf"(?i)\b({re.escape(w)})", r"<mark>\1</mark>", safe)
    return safe


@st.cache_data(show_spinner=False, max_entries=20)
def detect(name, data):
    d = rag.detect_profile(name, data)
    d.pop("signals", None)
    return d


def short(title, n=46):
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def where(chunk, preposition=False):
    page = chunk.get("page")
    sec = chunk.get("section") or ""
    if chunk.get("kind") == "footnote":
        text = f"endnote {chunk.get('note')}"
        return ("in " if preposition else "") + text
    if sec and not sec.lower().startswith("page") and sec != chunk.get("chapter"):
        text = f"{sec}, page {page}" if page else sec
        return ("in " if preposition else "") + text
    text = f"page {page}" if page else short(chunk.get("title", "the document"), 30)
    return ("on " if preposition else "") + text


# ---------------------------------------------------------------------------
# Sidebar: settings only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    model = st.selectbox("Chat model", CHAT_MODELS + ["Other…"],
                         help="Flash-lite is faster and cheaper. Flash gives fuller answers.")
    if model == "Other…":
        model = st.text_input("Model name", value="gemini-3.5-flash")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    k = st.slider("Sources per answer", 2, 10, 6)
    st.markdown("**Pipeline steps**")
    use_hybrid = st.toggle("Hybrid search (meaning + exact words)", value=True)
    use_routing = st.toggle("Section-aware routing", value=True, help="Used for long, structured documents.")
    use_rerank = st.toggle("Rerank with Gemini", value=True)
    use_expand = st.toggle("Add neighbouring passages and footnotes", value=True)
    use_rewrite = st.toggle("Understand follow-up questions", value=True)
    st.caption("Files are sent to Google Gemini to be processed. Don't upload confidential or client data.")

settings = dict(k=k, use_routing=use_routing, use_hybrid=use_hybrid, use_rerank=use_rerank, use_expand=use_expand)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="m-word">Margin</div><p class="m-tag">Ask questions about your documents. Every answer '
            'points to the passage it came from.</p>', unsafe_allow_html=True)

if not API_KEY:
    st.markdown(mascot.render("confused", "I need a Gemini API key before I can read anything. Add GOOGLE_API_KEY "
                              "under Settings → Secrets in Streamlit Cloud.", "hero"), unsafe_allow_html=True)
    st.stop()

kb = rag._MEMORY.get(ss.kb_key) if ss.kb_key else None
if kb is None and ss.entries and not ss.editing:
    # The server restarted or memory was cleared: rebuild quietly (usually instant from the disk cache).
    try:
        kb = rag.build_kb(ss.entries, API_KEY, model)
        ss.kb_key = kb["key"]
    except Exception:
        ss.entries = None

# ---------------------------------------------------------------------------
# Setup: upload, detect, read
# ---------------------------------------------------------------------------

if kb is None or ss.editing:
    pip_slot = st.empty()
    st.markdown('<div class="m-hero-title">Drop a document. Pip works out what kind it is.</div>',
                unsafe_allow_html=True)
    files = st.file_uploader("PDF, Word, Markdown or text", type=["pdf", "docx", "md", "txt"],
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_n}",
                             label_visibility="collapsed")

    if not files:
        pip_slot.markdown(mascot.render("greet", "Hi, I'm Pip! Drop a report, handbook or spec. I'll check whether "
                                        "it's plain text, a long structured report, or full of charts and tables, "
                                        "and read it the right way.", "hero"), unsafe_allow_html=True)
        if ss.editing and kb is not None and st.button("Back to the chat"):
            ss.editing = False
            st.rerun()
        st.stop()

    pending = []
    with st.spinner("Pip is looking at the pages…"):
        for f in files:
            data = f.getvalue()
            d = detect(f.name, data)
            pending.append({"name": f.name, "data": data, **d})

    for i, e in enumerate(pending):
        badge = f'<span class="m-badge {e["profile"]}">{TYPE_SHORT[e["profile"]]}</span>'
        known = ""
        if e.get("known"):
            known = f'<span class="m-chip">Hands-on {rag.USE_CASES[e["known"]]["label"].split(":")[0].lower()}</span>'
        chips = "".join(f'<span class="m-chip">{html.escape(r)}</span>' for r in e["reasons"])
        st.markdown(f'<div class="m-card"><h4>{html.escape(short(e["title"], 70))}</h4>'
                    f'<p>{badge}{known}{html.escape(e["name"])}</p><div>{chips}</div></div>', unsafe_allow_html=True)
        st.markdown('<p class="m-label">Not quite right? Change the type</p>', unsafe_allow_html=True)
        choice = st.selectbox("Document type", list(rag.PROFILES), index=list(rag.PROFILES).index(e["profile"]),
                              format_func=lambda p: rag.PROFILES[p]["label"],
                              key=f"type_{hashlib.md5(e['data']).hexdigest()}", label_visibility="collapsed")
        e["profile"] = choice

    if len(pending) == 1:
        line = TYPE_LINE[pending[0]["profile"]]
    else:
        counts = pd.Series([TYPE_SHORT[e["profile"]].lower() for e in pending]).value_counts()
        line = (f"{len(pending)} documents: " + ", ".join(f"{v} {t}" for t, v in counts.items())
                + ". I'll use the right strategy for each one.")
    pages = 0
    for e in pending:
        m = re.match(r"(\d+) pages", e["reasons"][0] if e["reasons"] else "")
        pages += int(m.group(1)) if m else 0
    if pages > 100:
        line += " It's a big read, so give me a few minutes."
    pip_slot.markdown(mascot.render("idle", line + " Change the type below if I got it wrong.", "hero"),
                      unsafe_allow_html=True)

    cols = st.columns([1, 1] if ss.editing and kb is not None else [1])
    go = cols[0].button("Read documents", type="primary", width="stretch")
    if ss.editing and kb is not None and cols[1].button("Cancel", width="stretch"):
        ss.editing = False
        st.rerun()

    if go:
        bar = st.progress(0.0)
        pip_slot.empty()
        dock = st.empty()  # Pip moves to the corner so progress stays visible wherever the page is scrolled
        moods = {"parse": "reading", "vision": "reading", "embed": "thinking", "wait": "confused"}

        doing = {"narrative": "Going page by page and keeping each fact with its sentence.",
                 "structured": "Mapping its chapters, regions and boxes.",
                 "visual": "Reading the text first, then I'll study each page image."}
        by_name = {e["name"]: e["profile"] for e in pending}

        def progress(stage, message, fraction, file=None):
            if stage == "parse" and file:
                message = f"Reading {short(file, 32)}. {doing[by_name.get(file, 'narrative')]}"
            dock.markdown(mascot.render(moods.get(stage, "reading"), message, "dock"), unsafe_allow_html=True)
            bar.progress(min(max(fraction, 0.0), 1.0))

        entries = [{k2: e[k2] for k2 in ("name", "data", "profile", "title", "known")} for e in pending]
        try:
            kb = rag.build_kb(entries, API_KEY, model, progress=progress)
        except Exception as err:
            bar.empty()
            dock.empty()
            pip_slot.markdown(mascot.render("confused", str(err) if isinstance(err, ValueError) else friendly_error(err),
                                            "hero"), unsafe_allow_html=True)
            st.stop()
        if not kb.get("suggestions") and not any(e.get("known") for e in entries):
            progress("embed", "Almost done. Thinking of good first questions…", 0.98)
            kb["suggestions"] = rag.suggest_questions(kb, rag.make_llm(API_KEY, model, 0.4))
            rag.save_kb(kb)
        ss.entries, ss.kb_key, ss.editing = entries, kb["key"], False
        ss.messages, ss.last_trace = [], None
        n_visual = sum(1 for c in kb["chunks"] if c["kind"] == "visual")
        extra = f", including {n_visual} pages of charts and tables" if n_visual else ""
        ss.pip = ("happy", f"All read: {len(kb['chunks'])} passages{extra}. Ask me anything!", True)
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Ready: documents loaded
# ---------------------------------------------------------------------------

pip_slot = st.empty()


def pip(mood, message, autohide=False):
    ss.pip = (mood, message, autohide)
    pip_slot.markdown(mascot.render(mood, message, "dock", autohide), unsafe_allow_html=True)


if ss.pip:
    pip(*ss.pip)
else:
    pip("idle", f"Ask me anything about {short(kb['title'], 40)}.", True)

bar_html = "".join(f'<span class="m-badge {f["profile"]}">{TYPE_SHORT[f["profile"]]}</span>'
                   f'<span class="m-chip">{html.escape(short(f["title"], 50))}</span>' for f in kb["files"])
c1, c2 = st.columns([5, 1.4], vertical_alignment="center")
c1.markdown(f'<div class="m-bar">{bar_html}<span class="m-chip">{len(kb["chunks"])} passages</span></div>',
            unsafe_allow_html=True)
if c2.button("Change documents", width="stretch"):
    ss.editing, ss.uploader_n, ss.pip = True, ss.uploader_n + 1, None
    st.rerun()

known = [e["known"] for e in (ss.entries or []) if e.get("known")]


def run_pipeline(question, history, opts, stream_to=None, narrate=False):
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
        if trace["routes"]:
            pip("reading", f"Checking only {', '.join(short(r, 34) for r in trace['routes'])}, so nothing gets mixed up.")
        else:
            pip("talking", "Found some good passages. Writing the answer…")
    messages = rag.build_messages(kb, question, results, history)
    t = time.time()
    if stream_to is not None:
        if narrate and trace["routes"]:
            pip("talking", "Writing the answer…")

        def gen():
            for chunk in llm.stream(messages):
                text = rag.get_text(chunk)
                if text:
                    yield text
        answer = stream_to.write_stream(gen())
    else:
        answer = rag.get_text(rag.call_with_retry(lambda: llm.invoke(messages)))
    times["generate"] = time.time() - t
    trace.update(question=question, times=times, results=results, answer=answer,
                 prompt="\n\n".join(f"[{m.type}]\n{m.content}" for m in messages))
    return answer, results, trace


def render_sources(results, query, answer):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
    n_cited = sum(1 for i in range(1, len(results) + 1) if i in cited)
    with st.expander(f"Sources · {n_cited} cited of {len(results)}"):
        for i, r in enumerate(results, 1):
            c = kb["chunks"][r["id"]]
            cls = "m-src" + (" cited" if i in cited else "") + (" extra" if r["reason"] != "retrieved" else "")
            extra = "" if r["reason"] == "retrieved" else f" · added: {r['reason']}"
            st.markdown(f'<div class="{cls}"><div class="meta"><b>[{i}]</b> {html.escape(c["header"])} · '
                        f'{KIND[c["kind"]]} · match {r["score"]:.2f}{html.escape(extra)}</div>'
                        f'{highlight(c["text"], query)}</div>', unsafe_allow_html=True)


tab_chat, tab_how, tab_test = st.tabs(["Chat", "How it works", "Test it"])

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

with tab_chat:
    starters = []
    for key in known:
        starters += rag.USE_CASES[key]["starters"]
    starters = (starters or kb.get("suggestions") or ["Summarise the main topics in this document."])[:3]
    if not ss.messages:
        st.caption("Try asking")
        with st.container(key="starters"):
            for i, q in enumerate(starters):
                if st.button(q, width="stretch", key=f"starter_{i}"):
                    ss.pending = q
                    st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("results"):
                render_sources(m["results"], m.get("query", ""), m["content"])

    question = st.chat_input("Ask about your documents") or ss.pop("pending", None)
    if question:
        history = list(ss.messages)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                answer, results, trace = run_pipeline(question, history, settings, stream_to=st, narrate=True)
            except Exception as e:
                pip("confused", friendly_error(e))
                st.stop()
            render_sources(results, trace["query"], answer)
        ss.messages += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer, "results": results, "query": trace["query"]}]
        ss.last_trace = trace
        if rag.NOT_FOUND.lower()[:20] in answer.lower():
            pip("confused", "I couldn't find that in your documents, so I won't guess. Try asking it another way?")
        else:
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
            top = results[cited[0] - 1] if cited and cited[0] <= len(results) else (results[0] if results else None)
            spot = f" {where(kb['chunks'][top['id']], preposition=True)}" if top else ""
            pip("happy", f"Found it{spot}. The exact passages are under my answer.", True)
        st.rerun()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tab_how:
    st.caption("What Pip did with each document, and how your last question was answered.")
    for f in kb["files"]:
        p = rag.PROFILES[f["profile"]]
        steps = "".join(f"<p><b style='color:var(--text)'>{html.escape(n)}.</b> {html.escape(t)}</p>"
                        for n, t in p["strategy"])
        st.markdown(f'<div class="m-card"><h4><span class="m-badge {f["profile"]}">{TYPE_SHORT[f["profile"]]}</span>'
                    f'{html.escape(short(f["title"], 60))}</h4>{steps}</div>', unsafe_allow_html=True)

    st.markdown('<div class="m-step">1. Parse and chunk</div>', unsafe_allow_html=True)
    kinds = pd.Series([c["kind"] for c in kb["chunks"]]).value_counts().to_dict()
    sizes = [len(c["text"]) for c in kb["chunks"]]
    st.write(f"**{len(kb['chunks'])}** passages, averaging {int(np.mean(sizes))} characters: "
             + ", ".join(f"{v} {KIND[k2]}" for k2, v in kinds.items()) + ".")
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
             "stored in an in-memory vector store alongside a BM25 keyword index. This passage starts with:")
    st.code(", ".join(f"{x:.4f}" for x in kb["vectors"][chosen["id"]][:8]) + ", …")

    trace = ss.last_trace
    st.markdown('<div class="m-step">3. Retrieve, rerank and assemble context</div>', unsafe_allow_html=True)
    if not trace:
        st.info("Ask a question in the Chat tab to see how it was answered.")
    else:
        st.write(f"Question: **{trace['question']}**")
        if trace["query"] != trace["question"]:
            st.write(f"Rewritten for search: **{trace['query']}**")
        if trace.get("routes"):
            st.write("Search limited to: **" + "**, **".join(trace["routes"]) + "**")

        def label(cid):
            return short(where(kb["chunks"][cid]), 40)

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
# Test it
# ---------------------------------------------------------------------------

COLS = ["Question", "Must contain", "Expected section", "Should say it doesn't know"]
BASIC = dict(k=4, use_routing=False, use_hybrid=False, use_rerank=False, use_expand=False)

with tab_test:
    defaults = [t for key in known for t in rag.USE_CASES[key]["tests"]]
    if defaults:
        st.caption("Pip recognised the hands-on task document and loaded its sample questions with expected answers.")
    else:
        defaults = [(q, "", "", False) for q in kb.get("suggestions", [])] + [
            ("What is the capital of France?", "", "", True)]
        st.caption("Add the facts each answer must contain to turn these into real checks.")
    st.caption("Must contain: separate required items with ; and alternatives with |  (e.g. 346; 45; 50). "
               "Expected section is optional and checks retrieval found the right part of the document.")
    cases = st.data_editor(pd.DataFrame(defaults, columns=COLS), num_rows="dynamic", width="stretch",
                           hide_index=True, key=f"tests_{kb['key']}")
    mode = st.radio("Pipeline to test", ["Advanced (current settings)", "Compare with basic RAG"], horizontal=True,
                    help="Basic = meaning search only, top 4, no routing, reranking or context expansion.")
    if st.button("Run tests", type="primary"):
        variants = [("Advanced", settings)] + ([("Basic", BASIC)] if mode.startswith("Compare") else [])
        todo = [r for _, r in cases.iterrows() if str(r["Question"] or "").strip()]
        rows, total, done = [], len(todo) * len(variants), 0
        bar_t = st.progress(0.0, text="Running tests…")
        pip("thinking", f"Running {total} tests. I'll mark my own homework.")
        for vname, opts in variants:
            for r in todo:
                q = str(r["Question"]).strip()
                try:
                    t = time.time()
                    answer, results, _ = run_pipeline(q, [], opts)
                    ok, why = rag.check_answer(answer, str(r["Must contain"] or ""),
                                               bool(r["Should say it doesn't know"]))
                    ret = rag.check_retrieval(kb, results, str(r["Expected section"] or ""))
                    rows.append({"Pipeline": vname, "Result": "✅ Pass" if ok else "❌ Fail", "Question": q, "Why": why,
                                 "Right section": "—" if ret is None else ("✅" if ret else "❌"),
                                 "Top source": where(kb["chunks"][results[0]["id"]]) if results else "",
                                 "Seconds": round(time.time() - t, 1), "Answer": answer})
                except Exception as e:
                    rows.append({"Pipeline": vname, "Result": "⚠️ Error", "Question": q, "Why": friendly_error(e)})
                done += 1
                bar_t.progress(done / total, text=f"Ran {done} of {total}")
        bar_t.empty()
        df = pd.DataFrame(rows)
        mcols = st.columns(len(variants))
        scores = {}
        for col, (vname, _) in zip(mcols, variants):
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
