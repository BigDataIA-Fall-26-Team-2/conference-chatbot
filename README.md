# Brochure RAG chatbot

Answers questions using conference brochures. Retrieval uses a separate Chroma collection for each registered company, a company metadata filter, and a result validation check. Backend answers must remain grounded in the returned evidence.

## Setup

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install Tesseract separately. On macOS, leave `TESSERACT_CMD` blank to use the executable on PATH, or set its installed path. The example Windows path must be changed on other operating systems.

## Ingestion and indexing

```bash
python -m ingestion.ocr
python -m ingestion.chunk
python -m retrieval.index
```

The committed `data/processed/chunks.jsonl` can be indexed immediately without rerunning OCR. Indexing uses local Sentence Transformers (`BAAI/bge-small-en-v1.5` by default); the first run downloads the model and requires internet access. No LLM API key is needed for indexing or retrieval.

`CHROMA_DIR` defaults to `chroma_db` relative to the repository root, regardless of the working directory. `EMBED_MODEL` selects the embedding model. Indexing and retrieval must use the same model; mismatches are rejected. To change models, choose a fresh `CHROMA_DIR` and reindex.

```bash
python -m retrieval.retrieve barndoor_ai "What does the gateway enforce?" --k 4
python -m pytest tests/test_isolation.py -q
```

## Backend handoff

```python
from retrieval.index import index_chunks
from retrieval.retrieve import retrieve

# Indexing normally runs offline, not during each chat request.
# index_chunks(chunks)
evidence = retrieve(company_id="barndoor_ai", question="What does the gateway enforce?", k=6)
```

Each result contains `id`, `text`, `company_id`, `source_file`, `page`, `chunk_index`, and `score`. Scores are cosine similarity (higher is better), not confidence probabilities. Similarity alone does not establish that a question is answerable; the backend must check whether the evidence supports its answer and cite the returned file/page metadata.

Only exact IDs from `config/companies.yaml` are accepted. Missing/unknown IDs and empty questions raise `ValueError`; valid companies without indexed content return `[]`. There is no global-search fallback. The UI sends the explicitly selected company ID and current question. Each company has separate chat history.

`index_chunks` validates all records and upserts stable IDs, so repeat indexing does not duplicate chunks and changed text updates an existing ID. It does not delete omitted IDs. If rechunking removes or renames chunks, rebuild into a fresh `CHROMA_DIR` to avoid stale evidence. Run ingestion/indexing before serving queries.

## Add a brochure

Put PDFs in `data/brochures/<company_id>/`, register the company in `config/companies.yaml`, and rerun ingestion and indexing. Company IDs come from the directory/registry, not model inference.

## API and UI (teammates' components)

Once those implementations are ready:

```bash
uvicorn backend.main:app --reload
streamlit run frontend/streamlit_app.py
```

## Embedding reference

The BGE query prefix follows the [model author's retrieval instructions](https://huggingface.co/BAAI/bge-small-en-v1.5). It is applied to questions only, not brochure chunks.

## Retrieval handoff validation (2026-10-30)

Run the complete suite, including actual BGE embeddings, with:

```bash
RUN_RETRIEVAL_LIVE=1 python -m pytest -q
```

After the model is cached, add `HF_HUB_OFFLINE=1` to verify offline operation. The default `python -m pytest -q` skips the actual-model evaluation but runs the Chroma integration tests with deterministic embeddings.

Validated on the committed 31 chunks from five companies:

- Ten curated brochure questions all found expected supporting chunks within the top three results. These fixtures measure evidence retrieval, not generated answer accuracy or a comprehensive benchmark.
- Full text, source filename, page number, and other metadata round-trip unchanged.
- All chunks fit within the model input limit: maximum 268 tokens versus 512 supported.
- Company filters, post-retrieval checks, alternating concurrent company requests, and adversarial question wording preserve company scope.
- Repeat indexing, updates, batches exceeding 64 records, persistence in a separate Python process, malformed input, model mismatch, and retry after embedding failure are covered.

Dhruvi should import `retrieve` from `retrieval.retrieve`, pass an exact company ID from the registry, and use the returned evidence and source metadata. Keep the default `k=6`: the correct passage is not always ranked first. Catch `ValueError` as invalid client input; treat storage/model failures as service errors rather than silently searching a different company. For an async FastAPI endpoint, offload this synchronous retrieval call to a worker thread (or use a synchronous endpoint).

Remaining boundaries:

- The backend must implement answerability checks, citations, and brochure-only generation. An unsupported question still returns similar chunks; an empty result is not the only insufficient-evidence condition.
- Existing OCR noise remains in the chunks. These tests do not certify extraction accuracy against every PDF page.
- Indexing is an upsert, not an atomic full-corpus replacement. Use a fresh database directory after removing/rechunking source material; do not index concurrently with serving queries.
- Company IDs are a scoping mechanism, not user authorization. The registry alias `bond ai` currently maps to `barndoor_ai`; verify it before implementing alias-based company routing. This retriever accepts only exact IDs and does not use aliases.

## Local conference demo

Start these in two terminals from the repository root with the virtual environment activated:

```bash
# Terminal 1 — API (reads your private .env)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit
python -m streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. Choose a company, type a question, and press Enter. Answers include file/page citations and a download of the cited brochure. Each company's chat is separate. Each question is answered independently; include the subject in follow-ups. The question counter has no ten-question limit. Use Download session to preserve the demo history; history otherwise lasts only for the browser session and is lost on a server restart.

`API_URL` defaults to http://127.0.0.1:8000. API keys stay in the backend environment and are never requested by the UI. Brochure downloads are limited to files within the selected company's directory. Failed requests show an error and a manual retry button; requests are not automatically resubmitted.

Ten prepared questions with expected answer summaries are in `tests/demo_questions.yaml`. Automated UI tests simulate ten questions, verify company-separated history, clear-chat behavior, retries, missing evidence, and offline backend handling. Run `python -m pytest -q`.

After changing imported frontend modules (such as `frontend/api_client.py`), restart the Streamlit process as well as refreshing the browser. A page rerun can retain the previous imported function signature. Restart the API separately after backend changes. A server restart clears in-memory chat history; export the session first if needed.
