# Margin — ask your documents

A RAG chatbot built with **Streamlit + LangChain + Google Gemini**. Attach a PDF, Word, Markdown or text file in the message bar and ask questions. Margin works out what kind of document it is, reads it with the matching strategy, and answers with citations to the exact passages. **Pip**, the mascot, explains what's happening at every step.

**Document → Parsing → Chunking → Embedding → Retrieval → Reranking → Context → LLM → Evaluation**

---

## UX research summary

### Who uses it

| User | Goal | What they need |
|---|---|---|
| **Reviewer** (manager grading the RAG tasks) | Check that the three hands-on documents are answered correctly, quickly | Fast proof: recognised task documents, sample tests pre-loaded, visible pipeline |
| **Everyday colleague** (non-technical) | Get a trustworthy answer from a long document | One obvious place to start, plain language, sources they can check |
| **Builder** (the app's owner) | Tune and debug the pipeline within a free API quota | Settings, test runs, the exact prompt, and how much quota is left |

### Friction found in the previous version

1. Uploading was a separate screen from chatting, which added a mode switch before the first question.
2. Technical settings (temperature, top-k, pipeline toggles) filled the sidebar, competing with what most users came to do.
3. There was no chat history, so switching documents lost the conversation.
4. The API quota was invisible until it ran out, and then it failed with an error.
5. Sources showed every retrieved passage in full, which was too much to scan.

### Design principles applied (Nielsen's usability heuristics)

| Heuristic | Decision |
|---|---|
| Visibility of system status | Pip narrates each stage; a progress panel shows reading; the usage card shows what's left and when it refreshes |
| Match with the real world | Quota is shown as "≈ N questions left today", not raw tokens. Document types are described by what they look like ("charts, tables & footnotes") |
| User control and freedom | The detected type can be changed and the file re-read; documents can be removed; New chat starts fresh; earlier chats stay in the sidebar |
| Error prevention | Warnings appear before the limit runs out; the Test view shows a run's cost first and disables Run when the allowance is too low; no Gemini call is made when there's no document yet |
| Recognition rather than recall | Suggested questions after reading; chat history with search; type cards on the home screen |
| Aesthetic and minimalist design | Settings live behind one button; sources show cited excerpts first and list the rest compactly |
| Help users recover from errors | Every error says what happened and when to try again (e.g. "the per-minute limit frees up in 34s") |

### User flow

```
Home ──► Attach file(s) in the message bar (optionally with a question)
          │
          ▼
   Pip detects each file's type and explains why (badge + reasons)
          │
          ▼
   Reading progress (Pip in the corner, status panel in the chat)
          │
          ├─► Question was included? ──► Answer with citations ──► Pip: "Found it in <section>, page N"
          │
          └─► No question ──► 3 suggested questions to tap
                                   │
                                   ▼
                         Follow-up questions (context is remembered)

Sidebar at any time: New chat · Documents (change type / re-read / remove) · How it works · Test it · Recent chats
Top bar: questions left · Export chat · Settings
```

---

## Gemini usage meter

The Gemini API doesn't report remaining quota, so Margin counts its own calls (`usage.py`):

- **Per minute:** requests and input tokens in a rolling 60-second window, with a countdown until the window frees up.
- **Per day:** answer requests (chat model) and reading requests (embeddings), which **reset at midnight Pacific time**, as documented by Google.
- **Tokens:** read from Gemini's usage data on each response; embedding tokens are estimated.
- **Questions left:** remaining requests divided by the measured requests per question (about 3: rewrite, rerank, answer).

Counts are shared by everyone using the app (one key) and saved on the server, so they survive page reloads. Google applies limits per project, so other apps using the same key aren't counted. Check the exact limits for your key in AI Studio → Usage, and set them as secrets if they differ from the defaults:

```toml
GOOGLE_API_KEY = "your-key"
GEMINI_DAILY_REQUESTS = 1000        # chat model requests per day
GEMINI_REQUESTS_PER_MINUTE = 15
GEMINI_TOKENS_PER_MINUTE = 250000
GEMINI_EMBED_DAILY_REQUESTS = 1000
```

---

## Automatic document-type detection

| Signal | Narrative | Structured | Visual |
|---|---|---|---|
| Chapter headings on pages or in bookmarks | none | 2 or more | — |
| Length | any | 150+ pages also counts | 40 pages or fewer |
| Short label-like lines | — | — | 27%+ of lines |
| Endnotes or footnotes section | — | — | present |
| Measurements per page (%, kg, W…) | — | — | 4+ |
| Vector graphics per page | — | — | chart-heavy |

On the three hands-on documents, detection is correct even with renamed files: GPO → Narrative, World Bank GEP → Structured, Apple iMac → Visual. These documents are also recognised by title, so their sample questions and expected answers load into **Test it** automatically.

## The three strategies

| | Narrative | Structured | Visual |
|---|---|---|---|
| Parsing | Page by page, columns in reading order | Chapters from bookmarks; regions, boxes, sub-sections from page headings; headers, broken glyphs, bibliographies removed | Text layer + Gemini vision description of each page image + endnotes as separate items |
| Chunking | Recursive, 1,000 chars | Hierarchical, inside sections only, with section path | 700 chars; chart/table descriptions kept whole |
| Retrieval | Hybrid: embeddings + BM25, rank fusion | + section-aware routing | Hybrid |
| Reranking | Gemini reorders the top 15 | Same | Same |
| Context | Top-k | + neighbouring passages, 16,000-char budget | + linked endnotes |

Each file is indexed once and cached by its content. Adding a second document to a chat only reads the new one.

## Files

| File | What it does |
|---|---|
| `app.py` | Interface: sidebar, chat with attachments, Documents, How it works, Test it, usage and settings |
| `rag_engine.py` | Detection, parsing strategies, embeddings, retrieval, reranking, prompts, checks; every Gemini call is counted |
| `usage.py` | Usage tracking: minute and daily windows, reset countdowns, questions-left estimate |
| `mascot.py` | Pip: animated gradient-orb SVG with seven moods |
| `.streamlit/config.toml` | Light theme, 100 MB uploads |
