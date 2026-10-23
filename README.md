# Margin — ask your documents

An end-to-end RAG (Retrieval-Augmented Generation) chatbot. Upload PDFs, Word, Markdown or text files and ask questions about them. Every answer cites the passage it came from, and the bot says so plainly when the answer isn't in your documents.

Built with **Streamlit + LangChain + Google Gemini**, the same stack as the Practical Generative AI Workshop.

## The pipeline

| Step | What happens | Where in `app.py` |
|---|---|---|
| Upload | PDF, DOCX, MD, TXT, several at once | sidebar |
| Parse | PDFs page by page (for page citations), Markdown split by headings, Word incl. tables | `parse_file` |
| Chunk | `RecursiveCharacterTextSplitter`, adjustable size and overlap | `build_knowledge_base` |
| Embed | Gemini embedding model, auto-detected for the API key | `get_embeddings` |
| Store | LangChain `InMemoryVectorStore`, cached per document set | `build_knowledge_base` |
| Retrieve | Cosine similarity top-k, optional MMR for varied sources, follow-up questions rewritten into standalone queries | `rewrite_query`, `retrieve` |
| Generate | Gemini answers only from numbered sources, cites them as [1], refuses when the answer isn't there; short-term memory of the last 6 messages | `build_messages`, `stream_answer` |

## What makes it more than a basic demo

- **Grounded answers with visible evidence.** Sources open under each answer with matched words highlighted and the cited ones marked.
- **"How it works" tab.** Shows each pipeline stage live: pages parsed, chunk browser, the actual embedding numbers, similarity scores and the exact prompt sent to Gemini.
- **"Test it" tab.** A small evaluation suite: questions with expected answers plus out-of-scope questions the bot must refuse. Editable, with pass/fail results.
- **Follow-up understanding.** "And who approves that?" is rewritten into a full question before searching.
- **Production touches.** Streaming responses, cached indexing, friendly errors for quota or key problems, a chunk cap for free-tier limits, an optional access password, and no API key in the code.

## Run it

**Online (Streamlit Community Cloud):** deploy this repository with `app.py` as the main file and add this secret:

```
GOOGLE_API_KEY = "your-gemini-key"
APP_PASSWORD = "optional-password"
```

**Locally:** `pip install -r requirements.txt`, set `GOOGLE_API_KEY`, then `streamlit run app.py`.

## Sample data

`sample_docs/lumen_design_handbook.md` is a fictional design-system handbook, so the app can be tried without uploading anything.

## Limits and next steps

Documents live only for the session (a persistent vector DB such as Chroma or pgvector is the next step), scanned image-only PDFs aren't read (would need OCR), and evaluation uses keyword checks rather than an LLM judge.
