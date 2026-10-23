# Margin — Advanced RAG over three different kinds of document

Margin is a RAG chatbot built with **Streamlit + LangChain + Google Gemini**. This version solves the three hands-on use cases. Each document has a different shape, so each gets its own strategy, while the overall pipeline stays the same:

**Document → Parsing → Chunking → Embedding → Retrieval → Reranking → Context → LLM → Evaluation**

| | Use case 1: GPO Annual Report | Use case 2: World Bank GEP | Use case 3: Apple iMac report |
|---|---|---|---|
| Shape | 82 pages, narrative text | 200 pages, chapters, regions, boxes | 12 pages, charts, tables, footnotes |
| Parsing | Page by page (two-column pages read in order) | Chapters from PDF bookmarks; regions, boxes and sub-sections (Recent developments, Outlook, Risks) from page headings; running headers, broken glyphs and bibliographies removed | Text layer **plus** a Gemini vision description of each page image **plus** each endnote as its own item |
| Chunking | Recursive, 1,000 chars, 150 overlap | Hierarchical: split inside sections only, 1,400 chars; every chunk carries its path (e.g. *Ch. 2 › South Asia › Outlook*) which is embedded with it | Small, 700 chars; chart and table descriptions kept whole |
| Retrieval | Hybrid: embeddings + BM25 keywords, merged by reciprocal rank fusion | Same, plus **section-aware routing**: a question naming a region, chapter, box or the executive summary searches only that section | Same hybrid search |
| Reranking | Gemini reorders the top 15 candidates | Same | Same |
| Context | Top-k passages | Plus neighbouring passages from the same section, within a 16,000-character budget | Plus **footnote linking**: a passage marked "STAR.18" brings endnote 18 with it |

## Why each choice

- **GPO:** The facts sit in normal sentences, so a clean baseline works. Hybrid search matters here because questions use exact names and numbers (FDsys, FY 2018) that pure meaning-based search can rank too low.
- **World Bank:** Every region section talks about "growth forecasts", so meaning-based search alone mixes regions up. Section paths in every chunk, plus routing on region and chapter names, keep answers in the right section. The PDF bookmarks were also checked against the page headings: the "South Asia" bookmark points one page early (to the Middle East table), so headings are trusted over bookmarks.
- **iMac:** The text layer scatters chart labels (the 445 kg baseline sits next to "Non-air shipping"), and a rule-based table extractor (pdfplumber) treated chart bars as tables and dropped rows of the power table. Reading the page image with a vision model recovers numbers with their labels. Endnotes carry conditions (for example, which configuration a carbon figure applies to), so they are linked to the passages that cite them.

## Evaluation

The **Test it** tab runs each use case's sample questions plus extra checks, including questions the bot must refuse (such as "FY 2022" for a 2018 report). Each test checks two things:

- **Answer:** the required facts are present and cited, or the bot correctly says it doesn't know.
- **Right section:** the retrieved passages come from the expected chapter or region. This tells a retrieval failure apart from a generation failure.

**Compare with basic RAG** runs the same tests through a plain pipeline (meaning search only, top 4, no routing, reranking or context expansion), so the gain from each technique is visible.

## Files

| File | What it does |
|---|---|
| `app.py` | The interface: chat, "How it works" inspector, "Test it" evaluation |
| `rag_engine.py` | The pipeline: parsing profiles, chunking, embeddings, hybrid retrieval, reranking, context assembly, prompts, checks |
| `GPO-Anual-report.pdf`, `GEP-Jun-2026.pdf`, `iMac_PER_Oct2024.pdf` | The three documents |
| `margin_index_*.pkl.gz` (optional) | Saved indexes, downloaded from the app, so documents load instantly without new embedding calls |

## Running it

On Streamlit Community Cloud, set the secret `GOOGLE_API_KEY`. Locally: `pip install -r requirements.txt`, then `streamlit run app.py`.

The first time each document is opened, it is parsed and embedded (about a minute for the iMac report and several minutes for the World Bank report on a free Gemini key, which pauses automatically at rate limits). Download each index from the "How it works" tab and add it to the repository, so later loads are instant.

## Limits and next steps

The index lives in memory (a persistent vector database such as Chroma or pgvector is the next step). Answer checks use keywords rather than an LLM judge. Vision descriptions are only as accurate as the model reading the page, which is why the text layer is kept alongside them.
