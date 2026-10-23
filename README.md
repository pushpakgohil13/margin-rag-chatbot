# Margin — upload any document, get grounded answers

Margin is a RAG chatbot built with **Streamlit + LangChain + Google Gemini**. Upload a PDF, Word, Markdown or text file. Margin **detects what kind of document it is** and picks the matching strategy, and **Pip**, the mascot, explains what it's doing at every step.

**Document → Parsing → Chunking → Embedding → Retrieval → Reranking → Context → LLM → Evaluation**

## Automatic document-type detection

Every upload is measured before it is read (`detect_profile` in `rag_engine.py`):

| Signal | Narrative | Structured | Visual |
|---|---|---|---|
| Chapter headings ("Chapter 1", "Ch. 2") on pages or in bookmarks | none | 2 or more | — |
| Length | any | 150+ pages also counts | 40 pages or fewer |
| Short label-like lines | low | — | 27%+ of lines |
| Endnotes or footnotes section | — | — | present |
| Measurements per page (%, kg, W…) | — | — | 4+ |
| Vector graphics per page (charts) | — | — | chart-heavy |

A document is **structured** if it has chapter structure, **visual** if it is short and shows at least two visual signals, and **narrative** otherwise. The reasons are shown to the user, who can change the type before reading. On the three hands-on documents, detection is correct even when the files are renamed:

| Document | Detected | Because |
|---|---|---|
| GPO 2018 Annual Report | Narrative | 82 pages, 2,171 characters per page, no chapter structure (its 57 bookmarks are statement titles, not chapters) |
| World Bank GEP, June 2026 | Structured | 200 pages, chapter headings, 18 bookmarks |
| Apple iMac environmental report | Visual | 12 pages, 30% short labels, endnotes section, 6 measurements per page, chart-heavy |

Several files can be uploaded together. Each one is read with its own strategy and they're searched as one knowledge base. The three hands-on documents are also recognised by their title, so their sample questions and expected answers load automatically into the Test tab.

## The three strategies

| | Narrative | Structured | Visual |
|---|---|---|---|
| Parsing | Page by page, columns in reading order | Chapters from bookmarks; regions, boxes and sub-sections from page headings; running headers, broken glyphs and bibliographies removed | Text layer + Gemini vision description of every page image + each endnote as its own item |
| Chunking | Recursive, 1,000 chars, 150 overlap | Hierarchical, inside sections only, 1,400 chars; each chunk carries its path (Ch. 2 › South Asia › Outlook) | 700 chars; chart and table descriptions kept whole |
| Retrieval | Hybrid: embeddings + BM25, reciprocal rank fusion | Same + section-aware routing (a question naming a region or chapter searches only there) | Same hybrid search |
| Reranking | Gemini reorders the top 15 | Same | Same |
| Context | Top-k | + neighbouring passages in the same section, 16,000-char budget | + linked endnotes for footnote markers |

## Pip

Pip is an original animated SVG character (`mascot.py`) with seven moods: waving on the welcome screen, **reading** while parsing (holding a page, eyes scanning), **thinking** while embedding and searching, **talking** while the answer streams, **happy** when an answer is found (with where it was found), and **confused** when the answer isn't in the documents or something fails. Motion respects the reduced-motion setting.

## Evaluation

**Test it** runs questions with required facts (`a; b|c` means a and (b or c)), questions the bot must refuse, and an optional expected section that checks retrieval separately from the answer. **Compare with basic RAG** runs the same tests through a plain pipeline to show what each technique adds.

## Files

| File | What it does |
|---|---|
| `app.py` | Interface: upload and detection, chat, How it works, Test it |
| `rag_engine.py` | Detection, parsing strategies, embeddings, hybrid retrieval, reranking, context, prompts, checks |
| `mascot.py` | Pip: SVG, animations, speech bubble |
| `.streamlit/config.toml` | Dark theme and a 100 MB upload limit |

## Running

Set the secret `GOOGLE_API_KEY` on Streamlit Community Cloud. Locally: `pip install -r requirements.txt`, then `streamlit run app.py`. Indexes are cached on the server by file content, so re-uploading the same document is instant until the app restarts.
