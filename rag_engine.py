"""
rag_engine.py — the RAG pipeline behind Margin.

    Document -> Parsing -> Chunking -> Embedding -> Retrieval -> Reranking -> Context -> LLM -> Evaluation

The same pipeline runs for every document, but three "profiles" change how the early
stages behave, because documents with different shapes need different handling:

    narrative   (GPO annual report)      page-aware parsing, recursive chunking
    structured  (World Bank GEP, 200 p)  chapter/section/box detection, contextual chunk
                                         headers, section-aware routing, neighbour expansion
    visual      (Apple iMac report)      page-image descriptions from a vision model,
                                         endnotes split out and linked to the facts they qualify
"""

import base64
import gzip
import hashlib
import io
import json
import pickle
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np

PIPELINE_VERSION = "4.0"
APP_DIR = Path(__file__).parent
CACHE_DIR = Path(tempfile.gettempdir()) / "margin_cache"
import usage as _usage  # noqa: E402
USAGE = _usage.Usage(CACHE_DIR / "usage.json")
NOT_FOUND = "I couldn't find that in your documents."

# ---------------------------------------------------------------------------
# Profiles: how each document shape is handled
# ---------------------------------------------------------------------------

PROFILES = {
    "narrative": {
        "label": "Plain narrative text",
        "chunk_size": 1000, "chunk_overlap": 150, "k": 5,
        "routing": False, "neighbours": 0, "vision": False, "footnotes": False,
        "strategy": [
            ("Parsing", "Text is read page by page so every answer can cite its page. The two-column layout is read in reading order."),
            ("Chunking", "Recursive splitting at paragraph, then sentence boundaries: 1,000 characters with 150 overlap, so a fact and its surrounding sentence stay together."),
            ("Retrieval", "Hybrid search: meaning-based (embeddings) plus exact-word search (BM25), merged with reciprocal rank fusion. Exact numbers and names like FDsys are found even when the wording differs."),
            ("Reranking", "Gemini re-orders the top candidates by how directly they answer the question."),
        ],
    },
    "structured": {
        "label": "Long, structured report",
        "chunk_size": 1400, "chunk_overlap": 150, "k": 6,
        "routing": True, "neighbours": 1, "vision": False, "footnotes": False,
        "strategy": [
            ("Parsing", "Chapters come from the PDF bookmarks. Regions, boxes and sub-sections (Recent developments, Outlook, Risks) are detected from headings at the top of each page. Running headers and damaged characters are cleaned."),
            ("Hierarchical chunking", "Text is split inside each section, never across two, and every chunk is stamped with its path, e.g. Ch. 2 › South Asia › Outlook. That path is embedded with the text."),
            ("Section-aware retrieval", "If a question names a chapter, region, box or the executive summary, search is limited to that section, so South Asia questions can't pull Sub-Saharan Africa facts."),
            ("Context management", "Neighbouring chunks from the same section are added for fuller context, within a fixed size budget."),
        ],
    },
    "visual": {
        "label": "Visual: charts, tables, footnotes",
        "chunk_size": 700, "chunk_overlap": 100, "k": 6,
        "routing": False, "neighbours": 0, "vision": True, "footnotes": True,
        "strategy": [
            ("Parsing", "Three layers per page: the text layer, a vision-model description of every chart, table and icon call-out on the page image, and each endnote as its own item."),
            ("Why vision", "Rule-based table extractors often misread designed reports. On the iMac report one treated chart bars as tables and dropped rows of the power table. Reading the page image recovers numbers together with their labels."),
            ("Chunking", "Smaller chunks (700 characters) because facts are short and dense. Chart and table descriptions stay whole."),
            ("Footnote linking", "When a retrieved passage carries a footnote marker (like STAR.18), that endnote is added to the context so its conditions are not lost."),
        ],
    },
}

# ---------------------------------------------------------------------------
# The three hands-on use cases
# ---------------------------------------------------------------------------

USE_CASES = {
    "gpo": {
        "label": "Use case 1: GPO annual report",
        "file": "GPO-Anual-report.pdf", "title": "GPO 2018 Annual Report", "profile": "narrative",
        "match": ["government publishing office 2018 annual report", "gpo 2018 annual report"],
        "about": "82 pages of narrative text with facts written into sentences. The baseline case.",
        "starters": [
            "How many depository libraries did GPO staff visit in FY 2018?",
            "What new digital system replaced FDsys, and when was it fully deployed?",
            "How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?",
        ],
        "tests": [
            ("How many depository libraries did GPO staff visit in FY 2018?", "166", "", False),
            ("What new digital system replaced FDsys, and when was it fully deployed?", "govinfo; 2018", "", False),
            ("How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?", "375", "", False),
            ("How many tangible copies were distributed to depository libraries?", "802,834", "", False),
            ("How many depository libraries did GPO visit in FY 2022?", "", "", True),
            ("What is the population of Canada?", "", "", True),
        ],
    },
    "gep": {
        "label": "Use case 2: World Bank economic report",
        "file": "GEP-Jun-2026.pdf", "title": "Global Economic Prospects, June 2026", "profile": "structured",
        "match": ["global economic prospects"],
        "about": "200 pages in chapters, regions, boxes and a statistical annex. Structure matters.",
        "starters": [
            "What is the growth outlook for South Asia in this report?",
            "What does Chapter 3 say about sovereign debt in developing economies?",
            "According to the executive summary, what are the main risks to the global outlook?",
        ],
        "tests": [
            ("What is the growth outlook for South Asia in this report?", "6.3; 7", "South Asia", False),
            ("What does Chapter 3 say about sovereign debt in developing economies?", "interest rate|spread|yield", "Ch. 3", False),
            ("According to the executive summary, what are the main risks to the global outlook?", "energy|commodity|hostilities|escalation", "Executive Summary", False),
            ("What is the growth forecast for Sub-Saharan Africa in 2026?", "4", "Sub-Saharan Africa", False),
            ("What growth is projected for East Asia and Pacific in 2026?", "4.2", "East Asia and Pacific", False),
            ("What does the report forecast for Antarctica's economy?", "", "", True),
        ],
    },
    "imac": {
        "label": "Use case 3: Apple iMac environmental report",
        "file": "iMac_PER_Oct2024.pdf", "title": "iMac Product Environmental Report", "profile": "visual",
        "match": ["imac product environmental report"],
        "about": "12 short pages where facts sit in charts, tables, icon call-outs and footnotes.",
        "starters": [
            "How much less energy does iMac use compared to the ENERGY STAR requirement?",
            "What is the total carbon footprint of the iMac, and how much comes from product use versus manufacturing?",
            "What material is the iMac stand made from, and what percentage of it is recycled?",
        ],
        "tests": [
            ("How much less energy does iMac use compared to the ENERGY STAR requirement?", "58", "", False),
            ("What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?", "346; 45; 50", "", False),
            ("What material is the iMac stand made from, and what percentage of it is recycled?", "aluminum|aluminium; 100", "", False),
            ("What is the iMac's power consumption in sleep mode at 115V?", "0.57", "", False),
            ("What were the baseline emissions before reductions in the carbon chart?", "445", "", False),
            ("What is the screen size of the iMac?", "", "", True),
        ],
    },
}

SAMPLE_TESTS_HANDBOOK = [
    ("How long does a new component take from approval to release?", "10 business days|ten business days", "", False),
    ("What contrast ratio is required for body text?", "4.5:1", "", False),
    ("What is the company's parental leave policy?", "", "", True),
]

REGION_ALIASES = {
    "East Asia and Pacific": ["east asia", "eap"],
    "Europe and Central Asia": ["europe and central asia", "central asia", "eca"],
    "Latin America and the Caribbean": ["latin america", "caribbean", "lac"],
    "Middle East, North Africa, Afghanistan, and Pakistan": ["middle east", "north africa", "mna", "menaap", "afghanistan", "pakistan"],
    "South Asia": ["south asia", "sar"],
    "Sub-Saharan Africa": ["sub-saharan", "subsaharan", "ssa"],
}
SUBHEADINGS = ["Recent developments", "Outlook", "Risks", "Introduction", "Conclusion", "Conclusions",
               "Policy challenges", "Policy implications", "References", "Annex"]
ORDINALS = {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth", "6": "sixth"}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

GLYPH_FIXES = {"—ó<": "T", "Fﬂä": "T", "7T_": "T", "7'_": "T"}


def normalise(text):
    """Repair damaged glyphs and ligatures (ﬁ, ﬂ) so keyword search and embeddings see real words."""
    for bad, good in GLYPH_FIXES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_running_header(line):
    """Drop lines like 'C H A P T ER 2 G L O B A L E C ON O M I C ...' and bare page numbers."""
    s = line.strip()
    if not s:
        return False
    tokens = s.split()
    single = sum(1 for t in tokens if len(t) <= 2)
    if len(tokens) >= 8 and single / len(tokens) > 0.6:
        return True
    return bool(re.fullmatch(r"[ivxlc]{1,6}|\d{1,3}", s.lower()))


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[.,:][0-9]+)*", text.lower())


def split_text(text, size, overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)


def read_pdf_pages(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return reader, [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def clean_title(t):
    return re.sub(r"\s+", " ", (t or "").replace("--", "")).strip()


def make_chunk(text, header, kind="text", **meta):
    return {"text": text.strip(), "header": header, "kind": kind, **meta}


# ---------------------------------------------------------------------------
# Parsing + chunking, one function per profile
# ---------------------------------------------------------------------------

def chunk_narrative(name, title, data, size, overlap):
    """Baseline: page-aware parsing with recursive chunking."""
    if not name.lower().endswith(".pdf"):
        return chunk_plain_file(name, title, data, size, overlap)
    _, pages = read_pdf_pages(data)
    chunks = []
    for page_no, raw in pages:
        lines = [l for l in normalise(raw).split("\n") if not is_running_header(l)]
        text = "\n".join(lines).strip()
        if len(text) < 40:
            continue
        for piece in split_text(text, size, overlap):
            chunks.append(make_chunk(piece, f"{title} · page {page_no}", source=name, page=page_no,
                                     section=f"Page {page_no}"))
    return chunks


def chunk_plain_file(name, title, data, size, overlap):
    """DOCX, Markdown and text uploads. Markdown keeps each heading with its own section."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        lines = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        sections = [(None, "\n".join(lines))]
    else:
        text = data.decode("utf-8", errors="ignore")
        sections, current, buf = [], None, []
        for line in text.splitlines():
            if ext == "md" and line.lstrip().startswith("#"):
                if "".join(buf).strip():
                    sections.append((current, "\n".join(buf)))
                current, buf = line.lstrip("# ").strip(), [line.lstrip("# ").strip()]
            else:
                buf.append(line)
        if "".join(buf).strip():
            sections.append((current, "\n".join(buf)))
    chunks = []
    for sec, body in sections:
        for piece in split_text(normalise(body), size, overlap):
            header = f"{title} › {sec}" if sec else title
            chunks.append(make_chunk(piece, header, source=name, section=sec or title))
    return chunks


def chapter_ranges(reader, n_pages):
    """Top-level PDF bookmarks become chapters (boxes and tables of contents are skipped)."""
    entries = []
    try:
        for item in reader.outline:
            if isinstance(item, list):
                continue
            title = clean_title(item.title)
            page = reader.get_destination_page_number(item) + 1
            if "--" in (item.title or "") or title in REGION_ALIASES:
                continue  # region bookmarks are sections, found from page headings instead
            if title.lower().startswith(("box", "contents", "summary of contents")):
                continue
            entries.append((page, title))
    except Exception:
        return []
    entries.sort()
    return entries


def sub_section_titles(reader):
    """Second-level bookmarks (e.g. regions) are used as headings to look for in page text."""
    titles = []
    try:
        def walk(items, depth):
            for it in items:
                if isinstance(it, list):
                    walk(it, depth + 1)
                elif depth >= 1 or "--" in (it.title or ""):
                    titles.append(clean_title(it.title))
        walk(reader.outline, 0)
    except Exception:
        pass
    return titles


def chunk_structured(name, title, data, size, overlap):
    """Hierarchical, section-aware chunking for long reports."""
    if not name.lower().endswith(".pdf"):
        return chunk_plain_file(name, title, data, size, overlap)
    reader, pages = read_pdf_pages(data)
    chapters = chapter_ranges(reader, len(pages))
    section_names = sub_section_titles(reader) + list(REGION_ALIASES.keys())
    section_norm = {re.sub(r"\W+", " ", s.lower()).strip(): s for s in section_names if s}
    sub_norm = {s.lower(): s for s in SUBHEADINGS}

    chunks, chapter, section, subsection = [], "Front matter", None, None
    for page_no, raw in pages:
        # 1. chapter from bookmarks
        new_chapter = next((t for p, t in reversed(chapters) if p <= page_no), "Front matter")
        if new_chapter != chapter:
            chapter, section, subsection = new_chapter, None, None

        text = raw.replace("\r", "")
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join words split across lines
        lines = [l.strip() for l in normalise(text).split("\n") if not is_running_header(l)]
        lines = [l for l in lines if l]
        if sum(len(l) for l in lines) < 60:
            continue

        # 2. section (region) from a heading at the top of the page
        top = re.sub(r"\W+", " ", " ".join(lines[:3]).lower()).strip()
        in_body = chapter.lower().startswith(("ch", "chapter"))
        for key, original in (section_norm.items() if in_body else []):
            if top.startswith(key):
                section, subsection = original, None
                break

        # 3. boxes are tagged page by page
        box_match = re.search(r"BOX (\d+\.\d+)\s+([^\n(]+)", "\n".join(lines))
        box = f"Box {box_match.group(1)} {box_match.group(2).strip()}" if box_match else None

        # 4. split the page at sub-headings so chunks never straddle two of them
        segments, buf = [], []
        for line in lines:
            if line.lower().rstrip(".") in sub_norm and len(line) < 40:
                if buf:
                    segments.append((subsection, buf))
                subsection, buf = sub_norm[line.lower().rstrip(".")], []
            else:
                buf.append(line)
        if buf:
            segments.append((subsection, buf))

        for sub, seg_lines in segments:
            body = "\n".join(seg_lines).strip()
            if len(body) < 40 or sub == "References":
                continue  # bibliographies add noise to search and cost embedding quota
            path = [chapter] + ([box] if box else ([section] if section else [])) + ([sub] if sub and not box else [])
            header = f"{title} › " + " › ".join(path) + f" · page {page_no}"
            for piece in split_text(body, size, overlap):
                chunks.append(make_chunk(piece, header, source=name, page=page_no, chapter=chapter,
                                         section=box or section or chapter, subsection=sub))
    return chunks


def split_endnotes(text):
    """Turn the endnotes page into one item per note, and separate any table that follows them."""
    table = None
    m = re.search(r"\n\s*Mode\s*\n.*", text, flags=re.S)
    if m:
        table, text = m.group(0).strip(), text[:m.start()]
    text = re.sub(r"©.*?(?=\n\s*\d{1,2}\s?[A-Z])", "", text, flags=re.S)  # drop the copyright block
    # A line starting with a number only begins a new note if it is the next number in sequence,
    # so numbers inside a note ("18 different criteria") are not mistaken for new notes.
    notes, current, last = {}, None, 0
    for line in text.split("\n"):
        m = re.match(r"\s*(\d{1,2})\s?([A-Za-z].*)", line)
        if m and last < int(m.group(1)) <= last + 2:
            current, last = int(m.group(1)), int(m.group(1))
            notes[current] = m.group(2)
        elif current is not None:
            notes[current] += " " + line.strip()
    return {n: re.sub(r"\s+", " ", b).strip() for n, b in notes.items()}, table


FOOTNOTE_REF = re.compile(r"(?<=[A-Za-z.,)%])(\d{1,2})(?=\s|$|[,.;:])")


def chunk_visual(name, title, data, size, overlap):
    """Text layer + endnotes. Page-image descriptions are added later in describe_pages()."""
    if not name.lower().endswith(".pdf"):
        return chunk_plain_file(name, title, data, size, overlap)
    _, pages = read_pdf_pages(data)
    chunks, endnote_text = [], ""
    for page_no, raw in pages:
        text = normalise(raw)
        if "Endnotes" in text or endnote_text:
            endnote_text += "\n" + text.replace("Endnotes", "")
            continue
        lines = [l for l in text.split("\n") if not re.fullmatch(r".*\| Product Environmental Report \d+", l.strip())]
        body = "\n".join(lines)
        for piece in split_text(body, size, overlap):
            refs = sorted({int(x) for x in FOOTNOTE_REF.findall(piece)})
            chunks.append(make_chunk(piece, f"{title} · page {page_no}", source=name, page=page_no,
                                     section=f"Page {page_no}", footnote_refs=refs))
    if endnote_text:
        notes, table = split_endnotes(endnote_text)
        for n, body in sorted(notes.items()):
            chunks.append(make_chunk(f"Endnote {n}: {body}", f"{title} · endnote {n}", kind="footnote",
                                     source=name, section="Endnotes", note=n))
        if table:
            chunks.append(make_chunk(table, f"{title} · power consumption table (endnotes page)", kind="table",
                                     source=name, section="Endnotes"))
    return chunks


VISION_PROMPT = """You are converting one page of a PDF report into searchable text for a retrieval system.
Describe every fact on this page that sits in a chart, table, diagram, icon call-out, big number or label.
- For bar or stacked charts: name the chart, then list every bar and segment with its label and value.
- For tables: reproduce them as a Markdown table with all rows and columns.
- For call-outs: write each one as a full sentence (e.g. "100 percent of the packaging is fiber-based").
- Keep footnote numbers attached to the claims they mark, written like [footnote 7].
Do not interpret, summarise or add facts that are not visible. If the page is only body text, reply "TEXT ONLY"."""


def render_page_png(data, page_index, scale=1.6):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    image = pdf[page_index].render(scale=scale).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def describe_pages(name, title, data, llm, progress=None):
    """Vision parsing: ask Gemini to read charts and tables from each page image."""
    from langchain_core.messages import HumanMessage
    from pypdf import PdfReader
    n = len(PdfReader(io.BytesIO(data)).pages)
    chunks = []
    for i in range(n):
        if progress:
            progress(f"Reading charts and tables on page {i + 1} of {n}", (i + 1) / n)
        png = base64.b64encode(render_page_png(data, i)).decode()
        msg = HumanMessage(content=[{"type": "text", "text": VISION_PROMPT},
                                    {"type": "image_url", "image_url": f"data:image/png;base64,{png}"}])
        reply = call_with_retry(lambda: llm.invoke([msg]))
        text = get_text(reply).strip()
        if text and "TEXT ONLY" not in text[:20]:
            refs = sorted({int(x) for x in re.findall(r"footnote (\d{1,2})", text)})
            chunks.append(make_chunk(text, f"{title} · page {i + 1} · charts and tables (read from the page image)",
                                     kind="visual", source=name, page=i + 1, section=f"Page {i + 1}",
                                     footnote_refs=refs))
    return chunks



# ---------------------------------------------------------------------------
# Automatic document-type detection
# ---------------------------------------------------------------------------

CHAPTERISH = re.compile(r"^\s*(ch\.?|chapter|part|section)\s*\d+\b", re.I)


def pdf_signals(data):
    """Measure the things that tell document shapes apart."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    step = max(1, n // 50)  # sample up to ~50 pages so long reports are measured in a few seconds
    texts = [reader.pages[i].extract_text() or "" for i in range(0, n, step)]
    titles = []
    try:
        def walk(items):
            for it in items:
                walk(it) if isinstance(it, list) else titles.append(clean_title(it.title))
        walk(reader.outline)
    except Exception:
        pass
    lines = [l.strip() for t in texts for l in t.split("\n") if l.strip()]
    joined = "\n".join(texts)
    paths = images = sampled = 0
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_raw
        pdf = pdfium.PdfDocument(data)
        for i in range(0, n, max(1, n // 25)):
            sampled += 1
            for obj in pdf[i].get_objects(max_depth=2):
                if obj.type == pdfium_raw.FPDF_PAGEOBJ_PATH:
                    paths += 1
                elif obj.type == pdfium_raw.FPDF_PAGEOBJ_IMAGE:
                    images += 1
        pdf.close()
    except Exception:
        pass
    meta_title = ""
    try:
        meta_title = (reader.metadata.title or "") if reader.metadata else ""
    except Exception:
        pass
    return {
        "pages": n,
        "chars_per_page": sum(map(len, texts)) / max(len(texts), 1),
        "chapter_pages": step * sum(1 for t in texts if re.search(r"(?im)^\s*(chapter|part)\s+\d+\b", t[:300])),
        "chapter_bookmarks": sum(1 for t in titles if CHAPTERISH.match(t)),
        "bookmarks": len(titles),
        "short_lines": sum(1 for l in lines if len(l) < 25) / max(len(lines), 1),
        "notes_section": bool(re.search(r"(?im)^\s*(endnotes|footnotes|notes)\s*$", joined)),
        "units_per_page": len(re.findall(r"\d+(?:\.\d+)?\s?(?:%|percent|kg|kWh|W\b|g\b|mm\b)", joined)) / max(len(texts), 1),
        "graphics_per_page": (paths + 20 * images) / max(sampled, 1),
        "meta_title": meta_title,
        "first_page": (reader.pages[0].extract_text() or "")[:600] if n else "",
    }


def detect_profile(name, data):
    """Work out which of the three strategies fits a file. Returns profile, reasons and a display title."""
    ext = name.lower().rsplit(".", 1)[-1]
    stem = re.sub(r"[_\-]+", " ", name.rsplit(".", 1)[0]).strip()
    if ext != "pdf":
        text = data.decode("utf-8", errors="ignore") if ext in ("md", "txt") else ""
        headings = 0
        if ext == "md":
            headings = len(re.findall(r"(?m)^#{1,3} ", text))
        elif ext == "docx":
            import docx
            d = docx.Document(io.BytesIO(data))
            headings = sum(1 for p in d.paragraphs if (p.style.name or "").lower().startswith("heading"))
            text = "\n".join(p.text for p in d.paragraphs)
        if headings >= 8 and len(text) > 30000:
            return {"profile": "structured", "title": stem, "known": None,
                    "reasons": [f"{headings} headings", f"{len(text):,} characters"]}
        return {"profile": "narrative", "title": stem, "known": None,
                "reasons": [f"{headings} headings" if headings else "continuous text", f"{len(text):,} characters"]}

    s = pdf_signals(data)
    title = clean_title(s["meta_title"]) or stem
    haystack = (s["meta_title"] + " " + s["first_page"]).lower()
    known = next((k for k, uc in USE_CASES.items() if any(m in haystack for m in uc.get("match", []))), None)

    visual_hits = []
    if s["short_lines"] >= 0.27:
        visual_hits.append(f"{s['short_lines']:.0%} of lines are short labels")
    if s["notes_section"]:
        visual_hits.append("has an endnotes section")
    if s["units_per_page"] >= 4:
        visual_hits.append(f"{s['units_per_page']:.0f} measurements per page")
    if s["graphics_per_page"] >= 35:
        visual_hits.append("chart-heavy pages")

    if s["chapter_pages"] >= 2 or s["chapter_bookmarks"] >= 3 or s["pages"] >= 150:
        reasons = [f"{s['pages']} pages"]
        if s["chapter_pages"] or s["chapter_bookmarks"]:
            reasons.append(f"{max(s['chapter_pages'], s['chapter_bookmarks'])} chapter headings")
        if s["bookmarks"]:
            reasons.append(f"{s['bookmarks']} bookmarks")
        profile = "structured"
    elif s["pages"] <= 40 and len(visual_hits) >= 2:
        profile, reasons = "visual", [f"{s['pages']} pages"] + visual_hits
    else:
        profile = "narrative"
        reasons = [f"{s['pages']} pages", f"{s['chars_per_page']:,.0f} characters per page", "no chapter structure"]
    return {"profile": profile, "title": title, "known": known, "reasons": reasons, "signals": s}


CHUNKERS = {"narrative": chunk_narrative, "structured": chunk_structured, "visual": chunk_visual}

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def is_rate_limit(err):
    m = str(err)
    return "429" in m or "RESOURCE_EXHAUSTED" in m or "quota" in m.lower() or "rate" in m.lower()


def call_with_retry(fn, attempts=6, wait=None):
    """Free-tier Gemini keys hit per-minute limits; wait and try again instead of failing."""
    delay = 15
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_rate_limit(e) or attempt == attempts - 1:
                raise
            hinted = re.search(r"retry(?:Delay)?\D{0,10}(\d+)", str(e))
            pause = min(int(hinted.group(1)) + 2, 90) if hinted else delay
            if wait:
                wait(pause)
            time.sleep(pause)
            delay = min(delay * 2, 90)


def embedding_model_name(api_key):
    try:
        from google import genai
        names = [m.name for m in genai.Client(api_key=api_key).models.list()
                 if "embedContent" in (m.supported_actions or [])]
        preferred = [n for n in names if "gemini-embedding" in n]
        return (preferred or names or ["models/gemini-embedding-001"])[0]
    except Exception:
        return "models/gemini-embedding-001"


def _raw_embeddings(api_key, model_name):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    kwargs = {"model": model_name, "google_api_key": api_key}
    if "gemini-embedding" in model_name:
        kwargs["output_dimensionality"] = 768
    return GoogleGenerativeAIEmbeddings(**kwargs)


def _raw_llm(api_key, model, temperature, max_tokens):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens,
                                  google_api_key=api_key, max_retries=2)


def _usage_tokens(message, prompt, text):
    meta = getattr(message, "usage_metadata", None) or {}
    inp = meta.get("input_tokens") or len(str(prompt)) // 4
    out = meta.get("output_tokens") or len(text) // 4
    return inp, out


class TrackedEmbeddings:
    """Counts every embedding call so the usage meter stays accurate."""

    def __init__(self, inner):
        self.inner = inner

    def embed_documents(self, texts):
        try:
            out = self.inner.embed_documents(texts)
        except Exception as e:
            if is_rate_limit(e):
                USAGE.record_limit_hit("embed")
            raise
        USAGE.record_embed(len(texts), sum(len(t) for t in texts))
        return out

    def embed_query(self, text):
        try:
            out = self.inner.embed_query(text)
        except Exception as e:
            if is_rate_limit(e):
                USAGE.record_limit_hit("embed")
            raise
        USAGE.record_embed(1, len(text))
        return out


class TrackedLLM:
    """Counts every chat-model call and its tokens (from Gemini's usage metadata when available)."""

    def __init__(self, inner):
        self.inner = inner

    def invoke(self, prompt):
        try:
            reply = self.inner.invoke(prompt)
        except Exception as e:
            if is_rate_limit(e):
                USAGE.record_limit_hit("chat")
            raise
        USAGE.record_chat(*_usage_tokens(reply, prompt, get_text(reply)))
        return reply

    def stream(self, prompt):
        text, last = "", None
        try:
            for chunk in self.inner.stream(prompt):
                if getattr(chunk, "usage_metadata", None):
                    last = chunk
                text += get_text(chunk)
                yield chunk
        except Exception as e:
            if is_rate_limit(e):
                USAGE.record_limit_hit("chat")
            raise
        USAGE.record_chat(*_usage_tokens(last, prompt, text))


def make_embeddings(api_key, model_name):
    return TrackedEmbeddings(_raw_embeddings(api_key, model_name))


def make_llm(api_key, model, temperature=0.2, max_tokens=1200):
    return TrackedLLM(_raw_llm(api_key, model, temperature, max_tokens))


def normalise_rows(m):
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


# ---------------------------------------------------------------------------
# Building a knowledge base (with on-disk and in-memory caching)
# ---------------------------------------------------------------------------

_MEMORY = {}


def serialise(kb):
    keep = {k: v for k, v in kb.items() if k not in ("bm25",)}
    return gzip.compress(pickle.dumps(keep))


def file_key(entry, embed_model):
    h = hashlib.sha256(json.dumps([PIPELINE_VERSION, embed_model, entry["name"], entry["profile"]]).encode())
    h.update(hashlib.sha256(entry["data"]).digest())
    return h.hexdigest()[:20]


def build_file_index(entry, api_key, chat_model, embed_model, progress=None):
    """Parse, chunk and embed ONE file with its own strategy. Cached by content, name and type."""
    say = progress or (lambda *_, **__: None)
    key = file_key(entry, embed_model)
    if key in _MEMORY:
        return _MEMORY[key]
    cached = CACHE_DIR / f"file_{key}.pkl.gz"
    if cached.exists():
        try:
            idx = pickle.loads(gzip.decompress(cached.read_bytes()))
            idx["loaded_from"] = "server cache"
            _MEMORY[key] = idx
            return idx
        except Exception:
            pass

    p, name, timings = PROFILES[entry["profile"]], entry["name"], {}
    t = time.time()
    say("parse", f"Reading {name}", 0.02, file=name)
    chunks = CHUNKERS[entry["profile"]](name, entry["title"], entry["data"], p["chunk_size"], p["chunk_overlap"])
    if p["vision"] and name.lower().endswith(".pdf"):
        vision_llm = make_llm(api_key, chat_model, temperature=0.0, max_tokens=2000)
        chunks += describe_pages(name, entry["title"], entry["data"], vision_llm,
                                 progress=lambda msg, f: say("vision", msg, 0.05 + 0.25 * f, file=name))
    for c in chunks:
        c["profile"], c["title"] = entry["profile"], entry["title"]
    timings["parse_chunk"] = time.time() - t
    if not chunks:
        raise ValueError(f"No readable text was found in {name}. Scanned PDFs (photos of pages) aren't supported yet.")

    t = time.time()
    emb = make_embeddings(api_key, embed_model)
    texts = [f"{c['header']}\n{c['text']}" for c in chunks]
    vectors, batch = [], 32
    for start in range(0, len(texts), batch):
        frac = 0.3 + 0.7 * start / len(texts)
        say("embed", f"Turning passages {start + 1}–{min(start + batch, len(texts))} of {len(texts)} into vectors",
            frac, file=name)
        part = texts[start:start + batch]
        vectors.extend(call_with_retry(lambda: emb.embed_documents(part),
                                       wait=lambda s: say("wait", f"Gemini asked me to slow down. Back in {s}s",
                                                          frac, file=name)))
    timings["embed"] = time.time() - t
    idx = {"key": key, "file": {"name": name, "profile": entry["profile"], "title": entry["title"]},
           "chunks": chunks, "vectors": normalise_rows(vectors), "timings": timings, "loaded_from": "built just now"}
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(gzip.compress(pickle.dumps(idx)))
    except Exception:
        pass
    _MEMORY[key] = idx
    return idx


def build_kb(entries, api_key, chat_model, progress=None):
    """Combine the per-file indexes of a conversation's documents into one searchable knowledge base."""
    embed_model = embedding_model_name(api_key)
    parts = [build_file_index(e, api_key, chat_model, embed_model, progress) for e in entries]
    key = "kb_" + hashlib.sha256("".join(p["key"] for p in parts).encode()).hexdigest()[:20]
    if key in _MEMORY:
        return _MEMORY[key]
    chunks = [{**c} for p in parts for c in p["chunks"]]  # copies, so each file's own index stays untouched
    for i, c in enumerate(chunks):
        c["id"] = i
    kb = {"key": key, "title": " + ".join(p["file"]["title"] for p in parts),
          "profiles": sorted({p["file"]["profile"] for p in parts}), "chunks": chunks,
          "vectors": np.vstack([p["vectors"] for p in parts]), "embed_model": embed_model,
          "files": [p["file"] for p in parts],
          "timings": {k: sum(p["timings"].get(k, 0) for p in parts) for k in ("parse_chunk", "embed")},
          "loaded_from": ", ".join(sorted({p["loaded_from"] for p in parts})), "suggestions": []}
    kb["bm25"] = make_bm25(chunks)
    _MEMORY[key] = kb
    return kb


def save_kb(kb):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{kb['key']}.pkl.gz").write_bytes(serialise(kb))
    except Exception:
        pass


SUGGEST_PROMPT = """Here are passages from a document. Write 3 short, specific questions a reader could ask
that these passages clearly answer. Vary them (a number, a definition or rule, a summary).
Return only the 3 questions, one per line, without numbering.

{passages}"""


def suggest_questions(kb, llm):
    """Starter questions for a document we haven't seen before."""
    pool = [c for c in kb["chunks"] if c["kind"] in ("text", "visual") and len(c["text"]) > 300]
    if not pool:
        return []
    picks = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(8, len(pool)))]
    passages = "\n\n".join(f"({c['header']})\n{c['text'][:500]}" for c in picks)
    try:
        reply = get_text(call_with_retry(lambda: llm.invoke(SUGGEST_PROMPT.format(passages=passages))))
        qs = [re.sub(r"^[\-\d.)\s]+", "", l).strip() for l in reply.split("\n")]
        return [q for q in qs if q.endswith("?")][:3]
    except Exception:
        return []


def make_bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenize(c["header"] + " " + c["text"]) for c in chunks])


# ---------------------------------------------------------------------------
# Retrieval: routing -> hybrid search -> fusion -> reranking -> context assembly
# ---------------------------------------------------------------------------


def has_profile(kb, name):
    return name in kb.get("profiles", [kb.get("profile")])


def section_aliases(kb):
    """Map every chapter, region, box and summary in the document to the phrases people use for it."""
    names = {}
    for c in kb["chunks"]:
        for field in ("chapter", "section"):
            if c.get(field):
                names[c[field]] = field
    aliases = {}
    for name, field in names.items():
        low = name.lower()
        terms = {low}
        ch = re.match(r"ch\.?\s*(\d+)", low)
        if ch:
            n = ch.group(1)
            terms |= {f"chapter {n}", f"ch. {n}", f"ch {n}", f"{ORDINALS.get(n, n)} chapter"}
        bx = re.match(r"box (\d+\.\d+)", low)
        if bx:
            terms = {f"box {bx.group(1)}"}
        if low.startswith("statistical"):
            terms |= {"statistical annex", "statistical appendix"}
        for region, alias in REGION_ALIASES.items():
            if region.lower() == low:
                terms |= set(alias)
        aliases[name] = (field, terms)
    return aliases


def route(kb, query):
    """Return the sections a question explicitly asks about (empty list = search everything)."""
    if not has_profile(kb, "structured"):
        return []
    q = " " + re.sub(r"[^\w.\- ]", " ", query.lower()) + " "
    hits = []
    for name, (field, terms) in section_aliases(kb).items():
        if name.lower() in ("front matter", "endnotes") or name.lower().startswith("page "):
            continue
        if any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", q) for t in terms if len(t) >= 3):
            hits.append((field, name))
    # "north africa" shouldn't also route to Sub-Saharan Africa and vice versa: keep the most specific
    return hits


def search(kb, query, api_key, k=6, use_routing=True, use_hybrid=True, use_rerank=True, use_expand=True,
           llm=None):
    trace = {"query": query}
    chunks, n = kb["chunks"], len(kb["chunks"])

    # 1. section-aware routing
    routes = route(kb, query) if use_routing else []
    allowed = np.ones(n, dtype=bool)
    if routes:
        mask = np.array([any(c.get(f) == name for f, name in routes) for c in chunks])
        if mask.sum() >= 2:
            allowed = mask
        else:
            routes = []
    trace["routes"] = [name for _, name in routes]

    # 2. dense search (meaning)
    emb = make_embeddings(api_key, kb["embed_model"])
    qv = normalise_rows([call_with_retry(lambda: emb.embed_query(query))])[0]
    sims = kb["vectors"] @ qv
    sims_masked = np.where(allowed, sims, -1.0)
    pool = 25
    dense_ids = [int(i) for i in np.argsort(-sims_masked)[:pool] if allowed[i]]

    # 3. keyword search (exact words and numbers)
    sparse_ids = []
    if use_hybrid:
        scores = np.asarray(kb["bm25"].get_scores(tokenize(query)))
        scores = np.where(allowed, scores, -1.0)
        sparse_ids = [int(i) for i in np.argsort(-scores)[:pool] if allowed[i] and scores[i] > 0]

    # 4. reciprocal rank fusion
    fused = {}
    for ranking in (dense_ids, sparse_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    candidates = sorted(fused, key=lambda c: -fused[c])[:15]
    trace["dense"] = dense_ids[:8]
    trace["sparse"] = sparse_ids[:8]
    trace["fused"] = candidates[:10]

    # 5. reranking with the LLM
    chosen = candidates[:k]
    trace["reranked"] = False
    if use_rerank and llm is not None and len(candidates) > 1:
        try:
            chosen = rerank(llm, query, candidates, chunks, k)
            trace["reranked"] = True
        except Exception:
            chosen = candidates[:k]

    # 6. context assembly: neighbours and linked footnotes, within a budget
    results = [{"id": c, "reason": "retrieved", "score": float(sims[c])} for c in chosen]
    budget, used = 16000, sum(len(chunks[c]["text"]) for c in chosen)
    have = set(chosen)
    if use_expand:
        extras = []
        for r in results:
            c = chunks[r["id"]]
            profile = PROFILES[c.get("profile", "narrative")]
            for note in c.get("footnote_refs", []) if profile["footnotes"] else []:
                fid = next((x["id"] for x in chunks if x.get("note") == note and x.get("source") == c.get("source")), None)
                if fid is not None and fid not in have:
                    extras.append({"id": fid, "reason": f"endnote {note}, referenced by a retrieved passage",
                                   "score": float(sims[fid])})
                    have.add(fid)
            for step in range(1, profile["neighbours"] + 1):
                for nb in (r["id"] - step, r["id"] + step):
                    if 0 <= nb < n and nb not in have and chunks[nb].get("section") == c.get("section"):
                        extras.append({"id": nb, "reason": "neighbouring passage in the same section",
                                       "score": float(sims[nb])})
                        have.add(nb)
        for e in extras:
            if used + len(chunks[e["id"]]["text"]) > budget:
                break
            results.append(e)
            used += len(chunks[e["id"]]["text"])
    trace["context_chars"] = used
    return results, trace


RERANK_PROMPT = """Question: {question}

Candidate passages:
{passages}

Pick the passages that help answer the question, most useful first. Prefer passages that contain the
specific numbers, names or statements asked about, and passages from the section the question names.
Return only a JSON list of passage numbers, for example [3, 1, 7]. Return at most {k} numbers."""


def rerank(llm, question, candidates, chunks, k):
    passages = "\n\n".join(f"[{i}] ({chunks[c]['header']})\n{chunks[c]['text'][:600]}"
                           for i, c in enumerate(candidates))
    reply = get_text(call_with_retry(lambda: llm.invoke(
        RERANK_PROMPT.format(question=question, passages=passages, k=k))))
    picks = [int(x) for x in re.findall(r"\d+", reply.split("]")[0])]
    ordered = [candidates[i] for i in dict.fromkeys(picks) if 0 <= i < len(candidates)]
    if not ordered:
        raise ValueError("empty rerank")
    rest = [c for c in candidates if c not in ordered]
    return (ordered + rest)[:k]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

CONDENSE_PROMPT = """Rewrite the user's latest message as one standalone search query, including any
names, regions, chapters or topics it refers to from the conversation. Return only the query.

Conversation:
{history}

Latest message: {question}

Standalone query:"""

SYSTEM_PROMPT = """You are Margin, an assistant that answers questions using ONLY the numbered sources
below, taken from the user's documents ({title}).

Sources can be body text, tables, descriptions of charts read from page images, or endnotes.
Numbers in tables, charts and endnotes are facts like any other.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge and do not guess.
- Cite the source number in square brackets after each fact, like [1] or [2][3].
- Keep facts in their place: if a source belongs to one region, chapter or product configuration,
  don't apply it to another.
- If an endnote limits or qualifies a figure you use, mention that briefly.
- If the sources don't contain the answer, reply exactly: "{not_found}" Then add one short
  sentence suggesting a related question the documents can answer.
- Greetings and questions about earlier messages can be answered from the conversation itself.
- Lead with the direct answer in one or two sentences, then short bullet points only if they help."""


def rewrite_query(llm, question, history, turns=6):
    if not history:
        return question
    transcript = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-turns:])
    try:
        out = get_text(call_with_retry(lambda: llm.invoke(
            CONDENSE_PROMPT.format(history=transcript, question=question)))).strip().strip('"')
        return out or question
    except Exception:
        return question


def build_messages(kb, question, results, history, turns=6):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    kind_label = {"text": "text", "table": "table", "visual": "chart/table read from page image",
                  "footnote": "endnote"}
    sources = "\n\n".join(
        f"[{i}] ({kb['chunks'][r['id']]['header']}; {kind_label.get(kb['chunks'][r['id']]['kind'], 'text')})\n"
        f"{kb['chunks'][r['id']]['text']}" for i, r in enumerate(results, 1))
    msgs = [SystemMessage(content=SYSTEM_PROMPT.format(title=kb["title"], not_found=NOT_FOUND)
                          + "\n\nSOURCES:\n" + sources)]
    for m in history[-turns:]:
        msgs.append((HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]))
    msgs.append(HumanMessage(content=question))
    return msgs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def check_answer(answer, must, should_refuse):
    """must: 'a; b|c' means a AND (b OR c). Returns (passed, reason)."""
    low = answer.lower()
    refused = "couldn't find" in low or "could not find" in low
    if should_refuse:
        return refused, ("Correctly said it doesn't know" if refused else "Answered when it should have refused")
    if refused:
        return False, "Said it doesn't know"
    groups = [[a.strip().lower() for a in g.split("|") if a.strip()] for g in (must or "").split(";")]
    groups = [g for g in groups if g]
    missing = [g for g in groups if not any(re.search(rf"(?<![\d.]){re.escape(a)}(?![\d])", low) for a in g)]
    if missing:
        return False, "Missing: " + ", ".join(" or ".join(g) for g in missing)
    if not re.search(r"\[\d+\]", answer):
        return False, "No citation"
    return True, "Correct and cited"


def check_retrieval(kb, results, expected_section):
    if not expected_section:
        return None
    exp = expected_section.lower()
    return any(exp in kb["chunks"][r["id"]]["header"].lower() for r in results if r["reason"] == "retrieved")
