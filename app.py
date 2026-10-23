"""
Margin — ask your documents.  (Advanced RAG edition)

The interface. The pipeline itself lives in rag_engine.py:
    Document -> Parsing -> Chunking -> Embedding -> Retrieval -> Reranking -> Context -> LLM -> Evaluation
"""

import html
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rag_engine as rag

APP_DIR = Path(__file__).parent
CHAT_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
HANDBOOK_NAME = "lumen_design_handbook.md"
HANDBOOK_TEXT = '# Lumen Design System — Team Handbook\n\nThis handbook explains how the product design team builds, reviews and ships work using the Lumen design system. It is a fictional sample document used to demonstrate the Margin RAG chatbot.\n\n# About Lumen\n\nLumen is the shared design system used by all product squads. It contains foundations (color, type, spacing, motion), a component library in Figma, and matching React components maintained by the front-end platform team. The design system team has four people: one lead designer, two product designers and one design engineer. Lumen releases a new version on the first Monday of every month.\n\n# Color tokens\n\nNever use raw hex values in a design file. Always use a semantic color token such as color.text.primary or color.surface.raised. Brand colors are only used for marketing surfaces and must not appear inside the product UI. Every color token has a light-mode and a dark-mode value, and designers must check both modes before handoff.\n\n# Accessibility standards\n\nAll product screens must meet WCAG 2.2 level AA. Body text needs a contrast ratio of at least 4.5:1 against its background. Large text (24px and above, or 18.5px bold) and UI icons need at least 3:1. Interactive targets must be at least 44 by 44 pixels on touch devices. Every flow must be fully usable with a keyboard, and focus states must always be visible. Color must never be the only way information is shown; pair it with an icon, label or pattern.\n\n# Typography\n\nThe product typeface is Inter for UI text and JetBrains Mono for code. The type scale has seven steps: 12, 14, 16, 20, 24, 32 and 40 pixels. Body text is 16px with a line height of 1.5. Do not use more than three type sizes on a single screen.\n\n# Spacing and layout\n\nLumen uses a 4px base spacing unit. All padding and margins must be multiples of 4px, with the most common values being 8, 16, 24 and 32. Layouts use a 12-column grid on desktop with 24px gutters, an 8-column grid on tablet and a 4-column grid on mobile with 16px gutters. Maximum content width is 1280px.\n\n# Icons\n\nIcons come in three sizes: 16px, 20px and 24px. Use the 20px size by default. All icons are outlined with a 1.5px stroke. Custom icons must be submitted to the design system team for review before they are used in production.\n\n# Design critique\n\nDesign crits happen every Wednesday from 3:00 to 4:00 pm. Anyone can bring work at any fidelity. Book a slot in the #design-crit channel by Tuesday at 6 pm. Each slot is 20 minutes: 5 minutes of context, 10 minutes of feedback and 5 minutes to agree on next steps. Feedback should be framed around the user problem and the brief, not personal taste.\n\n# Requesting a new component\n\nIf no existing component fits, fill in the component request form in Notion. Include the problem, at least two real screens where it will be used, and a link to any research. The design system team reviews requests every Friday. The standard turnaround for a new component, from approval to release in Figma and code, is 10 business days. Urgent requests need sign-off from the head of design.\n\n# Figma file conventions\n\nName files using the pattern Squad / Feature / Stage, for example Payments / Refund flow / Explorations. Every file must have a cover page with the owner, status and last-updated date. Status labels are Exploring, In review, Ready for dev and Shipped. Detach components only as a last resort, and leave a comment explaining why.\n\n# Handoff to engineering\n\nA design is ready for development only when it passes the handoff checklist: all states are designed (empty, loading, error, success), both light and dark modes are checked, copy has been reviewed by the content designer, accessibility annotations are added, and the file status is set to Ready for dev. Designers attend the sprint kickoff for any feature they hand off.\n\n# Deprecating components\n\nWhen a component is replaced, it is marked as deprecated in Figma and code. Deprecated components are kept for two monthly releases and then removed. Squads get a migration guide and a reminder in the #lumen-updates channel one week before removal.\n\n# Research and testing\n\nEvery major feature needs at least five usability test sessions before release. Research notes and recordings are stored in the Research Hub in Dovetail. Participants are paid a 50 dollar voucher for a 45 minute session. Personal data from research must be deleted after 12 months.\n\n# Tools and access\n\nThe team uses Figma for design, Notion for documentation, Dovetail for research and Linear for tracking work. New designers get Figma editor access on their first day and Dovetail access after completing the research privacy training.\n'

SOURCES = {**{k: v["label"] for k, v in rag.USE_CASES.items()},
           "handbook": "Sample: Lumen design handbook", "upload": "Upload my own files"}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Margin · Ask your documents", page_icon="🖍️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button p {
  font-family: 'Figtree', system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Newsreader', Georgia, serif; letter-spacing: -0.01em; }
.margin-title { font-family: 'Newsreader', Georgia, serif !important; font-size: 2.6rem !important;
  font-weight: 600 !important; line-height: 1.1 !important; margin: 0 !important; letter-spacing: -0.01em; }
.margin-sub { opacity: .72; margin: .35rem 0 1rem 0; max-width: 64ch; }
.margin-source { border-left: 3px solid #4A63D9; padding: .55rem .9rem; margin: .5rem 0 .8rem 0;
  background: rgba(74, 99, 217, .06); border-radius: 0 6px 6px 0; line-height: 1.55; white-space: pre-wrap; }
.margin-source.cited { border-left-color: #E2B100; background: rgba(255, 214, 0, .10); }
.margin-source.extra { border-left-style: dashed; }
.margin-source .meta { font-size: .82rem; opacity: .78; margin-bottom: .3rem; white-space: normal; }
.margin-source mark { background: rgba(255, 214, 0, .45); color: inherit; padding: 0 .1em; border-radius: 2px; }
.margin-pill { display: inline-block; font-size: .78rem; padding: .1rem .55rem; border-radius: 999px;
  border: 1px solid rgba(128,128,128,.35); margin: 0 .35rem .35rem 0; }
.margin-step { font-family: 'Newsreader', Georgia, serif; font-size: 1.25rem; font-weight: 600; margin: 1.1rem 0 .2rem 0; }
</style>
""", unsafe_allow_html=True)


def read_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


password = read_secret("APP_PASSWORD")
if password and not st.session_state.get("unlocked"):
    st.markdown('<div class="margin-title">Margin</div>', unsafe_allow_html=True)
    entered = st.text_input("Enter the access password to continue", type="password")
    if entered == password:
        st.session_state.unlocked = True
        st.rerun()
    elif entered:
        st.error("That password is incorrect. Ask the app owner for the current one.")
    st.stop()

API_KEY = read_secret("GOOGLE_API_KEY")


def friendly_error(err):
    msg = str(err)
    if rag.is_rate_limit(err):
        return "The Gemini free-tier limit was reached. Wait a minute and try again."
    if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in the app's secrets."
    if "404" in msg or "not found" in msg.lower():
        return "That Gemini model isn't available for this key. Pick another model in Answer settings."
    return f"Something went wrong while talking to Gemini: {msg[:300]}"


STOP = set("a an and are as at be by can do does for from has have how i in is it its me my of on or should "
           "that the their this to was what when where which who why will with you your".split())


def highlight(text, query):
    safe = html.escape(text)
    words = {w for w in re.findall(r"[A-Za-z0-9.:]+", query.lower()) if len(w) > 2 and w not in STOP}
    for w in sorted(words, key=len, reverse=True):
        safe = re.sub(rf"(?i)\b({re.escape(w)})", r"<mark>\1</mark>", safe)
    return safe


KIND = {"text": "text", "table": "table", "visual": "chart/table from page image", "footnote": "endnote"}


def find_bundled(filename):
    for p in (APP_DIR / filename, APP_DIR / "docs" / filename):
        if p.exists():
            return p.read_bytes()
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Your documents")
    source = st.radio("Knowledge source", list(SOURCES), format_func=SOURCES.get, label_visibility="collapsed")

    uploads, upload_profile = [], "narrative"
    if source == "upload":
        uploads = st.file_uploader("PDF, Word, Markdown or text files", type=["pdf", "docx", "txt", "md"],
                                   accept_multiple_files=True)
        upload_profile = st.selectbox("What kind of document is it?", list(rag.PROFILES),
                                      format_func=lambda p: rag.PROFILES[p]["label"],
                                      help="Picks the parsing and chunking strategy.")
    st.caption("Files are sent to Google Gemini to be processed. Don't upload confidential or client data.")

    profile_name = rag.USE_CASES[source]["profile"] if source in rag.USE_CASES else (
        upload_profile if source == "upload" else "narrative")
    profile = rag.PROFILES[profile_name]

    with st.expander("Answer settings"):
        model = st.selectbox("Chat model", CHAT_MODELS + ["Other…"],
                             help="Flash-lite is faster and cheaper. Flash gives fuller answers.")
        if model == "Other…":
            model = st.text_input("Model name", value="gemini-3.5-flash")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        k = st.slider("Sources per answer (top-k)", 2, 10, profile["k"])

    with st.expander("Pipeline steps"):
        use_hybrid = st.toggle("Hybrid search (meaning + exact words)", value=True)
        use_routing = st.toggle("Section-aware routing", value=profile["routing"],
                                disabled=not profile["routing"],
                                help="Only for long, structured documents.")
        use_rerank = st.toggle("Rerank with Gemini", value=True, help="One extra Gemini call per question.")
        use_expand = st.toggle("Add neighbouring passages and footnotes", value=True)
        use_rewrite = st.toggle("Understand follow-up questions", value=True)

    size = overlap = None
    if source in ("upload", "handbook"):
        with st.expander("Chunking settings"):
            size = st.slider("Chunk size (characters)", 300, 2000, profile["chunk_size"], 100)
            overlap = st.slider("Overlap (characters)", 0, 400, profile["chunk_overlap"], 25)

    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages, st.session_state.last_trace = [], None
        st.rerun()

settings = dict(k=k, use_routing=use_routing, use_hybrid=use_hybrid, use_rerank=use_rerank,
                use_expand=use_expand)

# ---------------------------------------------------------------------------
# Header, document loading, index building
# ---------------------------------------------------------------------------

st.markdown('<div class="margin-title">Margin</div>', unsafe_allow_html=True)
st.markdown('<p class="margin-sub">Ask questions about your documents. Every answer points to the passage it '
            'came from, and says so when the answer isn\'t there.</p>', unsafe_allow_html=True)

if not API_KEY:
    st.error("Add your Gemini API key to start. In Streamlit Cloud open the app's Settings, then Secrets, "
             'and add the line GOOGLE_API_KEY = "your-key". Get a free key at aistudio.google.com.')
    st.stop()

if source in rag.USE_CASES:
    uc = rag.USE_CASES[source]
    data = find_bundled(uc["file"])
    if data is None:
        st.warning(f"**{uc['file']}** isn't in your GitHub repository yet. Upload it next to app.py to make this "
                   "use case load automatically, or add it here for this session.")
        tmp = st.file_uploader(f"Add {uc['file']}", type=["pdf"], key=f"missing_{source}")
        if not tmp:
            st.stop()
        data = tmp.getvalue()
    files, title, name_hint = [(uc["file"], data)], uc["title"], source
elif source == "handbook":
    files, title, name_hint = [(HANDBOOK_NAME, HANDBOOK_TEXT.encode())], "Lumen design handbook", "handbook"
else:
    if not uploads:
        st.info("Upload one or more files in the sidebar and choose what kind of document they are.")
        st.stop()
    files, title, name_hint = [(f.name, f.getvalue()) for f in uploads], ", ".join(f.name for f in uploads), "custom"

doc_key = (source, tuple(n for n, _ in files), profile_name, size, overlap)
if st.session_state.get("doc_key") != doc_key:
    st.session_state.doc_key = doc_key
    st.session_state.messages, st.session_state.last_trace = [], None

status = st.empty()
bar = st.empty()


def progress(message, fraction):
    status.caption(message)
    bar.progress(min(max(fraction, 0.0), 1.0))


try:
    kb = rag.build_kb(files, title, profile_name, API_KEY, model, size=size, overlap=overlap,
                      name_hint=name_hint, progress=progress)
except Exception as e:
    status.empty(); bar.empty()
    st.error(str(e) if isinstance(e, ValueError) else friendly_error(e))
    st.stop()
status.empty(); bar.empty()

kinds = pd.Series([c["kind"] for c in kb["chunks"]]).value_counts().to_dict()
pills = [f"{len(kb['chunks'])} chunks"] + [f"{v} {KIND[k2]}" for k2, v in kinds.items() if k2 != "text"]
pills += [profile["label"], f"index {kb['loaded_from']}"]
st.markdown("".join(f'<span class="margin-pill">{html.escape(p)}</span>' for p in pills), unsafe_allow_html=True)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_trace", None)


def llm_for(temp=None):
    return rag.make_llm(API_KEY, model, temperature if temp is None else temp)


def run_pipeline(question, history, opts, stream_to=None):
    """Rewrite -> retrieve -> rerank -> context -> generate. Returns answer, results, trace."""
    llm, times = llm_for(), {}
    t = time.time()
    query = rag.rewrite_query(llm_for(0.0), question, history) if (use_rewrite and history) else question
    times["rewrite"] = time.time() - t
    t = time.time()
    results, trace = rag.search(kb, query, API_KEY, llm=llm_for(0.0), **opts)
    times["retrieve"] = time.time() - t
    messages = rag.build_messages(kb, question, results, history)
    t = time.time()
    if stream_to is not None:
        def gen():
            for chunk in llm.stream(messages):
                text = rag.get_text(chunk)
                if text:
                    yield text
        answer = stream_to.write_stream(gen())
    else:
        answer = rag.get_text(rag.call_with_retry(lambda: llm.invoke(messages)))
    times["generate"] = time.time() - t
    trace.update(question=question, times=times, prompt="\n\n".join(f"[{m.type}]\n{m.content}" for m in messages),
                 results=results, answer=answer)
    return answer, results, trace


def render_sources(results, query, answer):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
    n_cited = sum(1 for i in range(1, len(results) + 1) if i in cited)
    with st.expander(f"Sources · {n_cited} cited of {len(results)} in context"):
        for i, r in enumerate(results, 1):
            c = kb["chunks"][r["id"]]
            classes = "margin-source" + (" cited" if i in cited else "") + (" extra" if r["reason"] != "retrieved" else "")
            extra = "" if r["reason"] == "retrieved" else f" · added: {r['reason']}"
            st.markdown(
                f'<div class="{classes}"><div class="meta"><b>[{i}]</b> {html.escape(c["header"])} · '
                f'{KIND[c["kind"]]} · match {r["score"]:.2f}{html.escape(extra)}'
                f'{" · used in answer" if i in cited else ""}</div>{highlight(c["text"], query)}</div>',
                unsafe_allow_html=True)


tab_chat, tab_how, tab_test = st.tabs(["Chat", "How it works", "Test it"])

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

with tab_chat:
    if source in rag.USE_CASES:
        st.caption(rag.USE_CASES[source]["about"])
    starters = rag.USE_CASES[source]["starters"] if source in rag.USE_CASES else (
        ["How long does a new component request take?", "What contrast ratio does body text need?"]
        if source == "handbook" else ["Summarise the main topics in these documents."])
    if not st.session_state.messages:
        cols = st.columns(len(starters))
        for col, q in zip(cols, starters):
            if col.button(q, width="stretch"):
                st.session_state.pending = q
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("results"):
                render_sources(m["results"], m.get("query", ""), m["content"])

    question = st.chat_input("Ask about your documents") or st.session_state.pop("pending", None)
    if question:
        history = list(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                answer, results, trace = run_pipeline(question, history, settings, stream_to=st)
            except Exception as e:
                st.error(friendly_error(e))
                st.stop()
            render_sources(results, trace["query"], answer)
        st.session_state.messages += [{"role": "user", "content": question},
                                       {"role": "assistant", "content": answer, "results": results,
                                        "query": trace["query"]}]
        st.session_state.last_trace = trace

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tab_how:
    st.caption(f"Strategy for this document: {profile['label'].lower()}. Each stage, using the index loaded now.")
    for name, text in profile["strategy"]:
        st.markdown(f"**{name}.** {text}")

    st.markdown('<div class="margin-step">1. Parse and chunk</div>', unsafe_allow_html=True)
    sizes = [len(c["text"]) for c in kb["chunks"]]
    st.write(f"**{len(kb['chunks'])}** chunks from {len(kb['files'])} file(s), target size {kb['chunk_size']} "
             f"characters with {kb['chunk_overlap']} overlap (average {int(np.mean(sizes))}).")
    if profile_name == "structured":
        rows = {}
        for c in kb["chunks"]:
            key = (c.get("chapter", ""), c.get("section", ""))
            r = rows.setdefault(key, {"Chapter": key[0], "Section": key[1], "Pages": [], "Chunks": 0})
            r["Pages"].append(c.get("page", 0)); r["Chunks"] += 1
        table = [{**r, "Pages": f"{min(r['Pages'])}–{max(r['Pages'])}"} for r in rows.values()]
        st.caption("Document map detected during parsing")
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    if kinds:
        st.caption("Chunk types: " + ", ".join(f"{v} {KIND[k2]}" for k2, v in kinds.items()))
    kind_filter = st.selectbox("Browse chunks of type", ["all"] + list(kinds), format_func=lambda x: KIND.get(x, x))
    pool = [c for c in kb["chunks"] if kind_filter == "all" or c["kind"] == kind_filter]
    ix = st.number_input(f"Chunk (1–{len(pool)})", 1, len(pool), 1) - 1
    chosen = pool[ix]
    st.markdown(f'<div class="margin-source"><div class="meta">{html.escape(chosen["header"])} · '
                f'{KIND[chosen["kind"]]}</div>{html.escape(chosen["text"])}</div>', unsafe_allow_html=True)

    st.markdown('<div class="margin-step">2. Embed and store</div>', unsafe_allow_html=True)
    st.write(f"Each chunk, with its header, became **{kb['vectors'].shape[1]}** numbers using `{kb['embed_model']}`. "
             "Vectors are kept in an in-memory NumPy store, and a BM25 keyword index is built alongside. "
             "First eight numbers of this chunk:")
    st.code(", ".join(f"{x:.4f}" for x in kb["vectors"][chosen["id"]][:8]) + ", …")
    st.download_button("Download this index", rag.serialise(kb), file_name=f"margin_index_{kb['name_hint']}.pkl.gz",
                       help="Upload this file to your GitHub repository next to app.py and this document will "
                            "load instantly, with no embedding calls.")

    trace = st.session_state.last_trace
    st.markdown('<div class="margin-step">3. Retrieve, rerank and assemble context</div>', unsafe_allow_html=True)
    if not trace:
        st.info("Ask a question in the Chat tab to see how it was answered.")
    else:
        st.write(f"Question: **{trace['question']}**")
        if trace["query"] != trace["question"]:
            st.write(f"Rewritten for search: **{trace['query']}**")
        if trace.get("routes"):
            st.write("Search limited to: **" + "**, **".join(trace["routes"]) + "**")

        def label(cid):
            return kb["chunks"][cid]["header"].split(" › ")[-1] if " › " in kb["chunks"][cid]["header"] else kb["chunks"][cid]["header"]

        c1, c2, c3 = st.columns(3)
        c1.caption("Meaning search (top 5)"); c1.write("\n".join(f"- {label(i)}" for i in trace["dense"][:5]) or "—")
        c2.caption("Keyword search (top 5)"); c2.write("\n".join(f"- {label(i)}" for i in trace["sparse"][:5]) or "off")
        c3.caption("After fusion" + (" and reranking" if trace["reranked"] else ""))
        c3.write("\n".join(f"- {label(r['id'])}" for r in trace["results"] if r["reason"] == "retrieved"))
        extras = [r for r in trace["results"] if r["reason"] != "retrieved"]
        if extras:
            st.write("Added for context: " + "; ".join(f"{label(r['id'])} ({r['reason']})" for r in extras))
        st.caption(f"Context sent to Gemini: {trace['context_chars']:,} characters. Retrieval took "
                   f"{trace['times']['retrieve']:.1f}s, the answer {trace['times']['generate']:.1f}s.")
        st.markdown('<div class="margin-step">4. Generate</div>', unsafe_allow_html=True)
        with st.expander("See the exact prompt sent to Gemini"):
            st.code(trace["prompt"], language=None)

# ---------------------------------------------------------------------------
# Test it
# ---------------------------------------------------------------------------

COLS = ["Question", "Must contain", "Expected section", "Should say it doesn't know"]
BASIC = dict(k=4, use_routing=False, use_hybrid=False, use_rerank=False, use_expand=False)

with tab_test:
    st.caption("Must contain: separate required items with ; and alternatives with |  (e.g. 346; 45; 50). "
               "Expected section is optional and checks that retrieval found the right part of the document.")
    defaults = rag.USE_CASES[source]["tests"] if source in rag.USE_CASES else (
        rag.SAMPLE_TESTS_HANDBOOK if source == "handbook" else [("What is this document about?", "", "", False)])
    cases = st.data_editor(pd.DataFrame(defaults, columns=COLS), num_rows="dynamic", width="stretch",
                           hide_index=True, key=f"tests_{source}")
    mode = st.radio("Pipeline to test", ["Advanced (current settings)", "Compare with basic RAG"], horizontal=True,
                    help="Basic = meaning search only, top 4, no routing, reranking or context expansion.")

    if st.button("Run tests", type="primary"):
        variants = [("Advanced", settings)] + ([("Basic", BASIC)] if mode.startswith("Compare") else [])
        rows, todo = [], [r for _, r in cases.iterrows() if str(r["Question"] or "").strip()]
        bar_t = st.progress(0.0, text="Running tests…")
        total, done = len(todo) * len(variants), 0
        for vname, opts in variants:
            for r in todo:
                q = str(r["Question"]).strip()
                try:
                    t = time.time()
                    answer, results, _ = run_pipeline(q, [], opts)
                    ok, why = rag.check_answer(answer, str(r["Must contain"] or ""), bool(r["Should say it doesn't know"]))
                    ret = rag.check_retrieval(kb, results, str(r["Expected section"] or ""))
                    rows.append({"Pipeline": vname, "Result": "✅ Pass" if ok else "❌ Fail", "Question": q,
                                 "Why": why, "Right section": "—" if ret is None else ("✅" if ret else "❌"),
                                 "Top source": kb["chunks"][results[0]["id"]]["header"].split(" · ")[0][-60:] + " · " +
                                               kb["chunks"][results[0]["id"]]["header"].split(" · ")[-1],
                                 "Seconds": round(time.time() - t, 1), "Answer": answer})
                except Exception as e:
                    rows.append({"Pipeline": vname, "Result": "⚠️ Error", "Question": q, "Why": friendly_error(e)})
                done += 1
                bar_t.progress(done / total, text=f"Ran {done} of {total}")
        bar_t.empty()
        df = pd.DataFrame(rows)
        cols = st.columns(len(variants))
        for col, (vname, _) in zip(cols, variants):
            part = df[df["Pipeline"] == vname]
            col.metric(f"{vname} pipeline", f"{(part['Result'] == '✅ Pass').sum()} of {len(part)} passed")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Fails show where to improve: a ❌ in Right section is a retrieval problem; a ✅ section with a "
                   "wrong answer is a generation problem.")
