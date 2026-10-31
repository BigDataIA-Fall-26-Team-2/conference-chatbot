# Conference Brochure RAG Chatbot: Team Plan

## Objective

Build a conference chatbot that answers questions using company brochures only. Use a real RAG pipeline: OCR extraction, chunking, embeddings, vector storage, company-filtered retrieval, and grounded answer generation with citations.

Do not load entire brochures into the model prompt. If the retrieved evidence does not answer a question, acknowledge that the brochures do not provide the information. Company A's answers must never use Company B's brochure content.

## Agreed stack

- OCR: Tesseract, with PDF pages converted to images before OCR.
- Backend: Python and FastAPI.
- Vector database: Chroma with persistent storage.
- Frontend: Streamlit.
- Embedding and chat models: use the team's agreed providers and model names, configured through environment variables. Use the same embedding model for indexing and querying.

## Shared prerequisite: initial setup led by Guna

Complete this setup before starting independent implementation:

1. Create the repository structure and initial commit.
2. Assign stable company IDs, such as `barndoor_ai`, and agree on a company registry.
3. Organize PDFs under `data/brochures/<company_id>/`.
4. Add `.env.example` with configuration names, without real credentials.
5. Ignore `.env`, local vector database files, and generated artifacts in Git.
6. Add sample chunk records and a mock chat response matching the contracts below.
7. Record installation and startup commands in `README.md`, including Tesseract and PDF rendering dependencies.

```text
ingestion/           # Guna
retrieval/           # Lokesh
backend/             # Dhruvi
frontend/            # Shravya
tests/               # Shared integration checks
data/
  brochures/         # PDFs organized by company_id
  samples/           # Sample chunks and mock API response
.env.example
.gitignore
README.md
plan.md
```

## Work allocation

### Guna: repository setup, OCR, and chunking

**Owns:** `ingestion/`, initial repository structure, and sample data.

- Make the initial commit so teammates can start in parallel.
- Convert PDF pages to images and extract text using Tesseract.
- Inspect OCR quality and clean noise while preserving headings and feature lists.
- Split content into meaningful chunks; start with roughly 400–700 tokens and modest overlap, adjusting based on brochure structure.
- Preserve page boundaries for accurate page citations.
- Attach company ID, filename, page number, and chunk index to each chunk.
- Generate stable chunk IDs and export standardized records.

**Interface:**

```python
ingest_folder(path: str) -> list[dict]
```

**Done when:** Every brochure produces readable chunks with correct company and page metadata, and important lists remain complete or retrievable across chunks.

### Lokesh: embeddings, vector storage, and retrieval

**Owns:** `retrieval/`.

- Generate embeddings for the standardized chunks.
- Store vectors, text, and metadata in persistent Chroma storage.
- Use stable IDs to avoid duplicates when reindexing unchanged chunks.
- Embed incoming questions with the same model used during indexing.
- Apply the selected `company_id` filter inside every vector database query.
- Return relevant chunks with original metadata and a retrieval score; document whether higher or lower scores are better.
- Validate retrieval quality using known brochure questions and paraphrases.

**Interfaces:**

```python
index_chunks(chunks: list[dict]) -> None
retrieve(company_id: str, question: str, k: int = 6) -> list[dict]
```

**Parallel starting point:** Use the shared sample chunks until Guna's real chunks are ready.

**Done when:** Indexes persist across restarts, repeated indexing avoids duplicate records, and searches return relevant evidence only from the selected company.

### Dhruvi: chatbot logic and backend API

**Owns:** `backend/` and backend integration.

- Implement `GET /companies` and `POST /chat` using FastAPI.
- Validate company IDs and reject empty questions.
- Connect Lokesh's retriever to the chat model.
- Provide only retrieved evidence as brochure context.
- Treat brochure text as evidence, not executable instructions.
- Require brochure-grounded answers and file/page citations.
- Return an insufficient-evidence response when the answer is unsupported.
- Validate that returned source references correspond to retrieved chunks.
- Handle model/API errors with clear responses.

**Interface:**

```python
answer_question(company_id: str, question: str) -> dict
```

**Parallel starting point:** Use a mock retriever returning the shared sample chunks.

**Done when:** The API returns grounded answers, valid citations, and clear missing-information responses. Each request uses the explicit company scope.

### Shravya: frontend and demo verification

**Owns:** `frontend/` and the demo question set.

- Build the Streamlit app with a company selector and chat input.
- Display answers and source filenames/page numbers.
- Add loading indicators and readable error states.
- Clear or separate chat history when the selected company changes.
- Connect the UI to Dhruvi's endpoints.
- Prepare questions and expected answers by checking the actual brochures.
- Run the final end-to-end demo with the team.

**Parallel starting point:** Use the shared mock API response until the backend is ready.

**Done when:** Users can select a company, ask questions, inspect citations, and switch companies without carrying another company's context into answers.

## Shared contracts

### Chunk record

Page numbers are one-based. Keep field names identical across ingestion, retrieval, and backend code.

```json
{
  "id": "barndoor_ai_4_Barndoor_p2_c1",
  "text": "Brochure content...",
  "company_id": "barndoor_ai",
  "source_file": "4_Barndoor.pdf",
  "page": 2,
  "chunk_index": 1
}
```

Retrieved chunks retain these fields and add `score`.

### Company list

`GET /companies` returns:

```json
{
  "companies": [
    {"company_id": "barndoor_ai", "name": "Barndoor.ai"}
  ]
}
```

### Chat request

`POST /chat` accepts:

```json
{
  "company_id": "barndoor_ai",
  "question": "What are its six features?"
}
```

### Chat response

```json
{
  "answer": "According to the brochure...",
  "sources": [
    {"source_file": "4_Barndoor.pdf", "page": 2}
  ],
  "insufficient_evidence": false
}
```

When evidence is missing, return a clear explanation with `insufficient_evidence: true`. Do not invent sources or use general model knowledge to fill gaps.

## Parallel execution and integration

1. **Shared setup:** Guna commits the structure, contracts, and sample fixtures. Everyone pulls the initial commit and works on a separate branch.
2. **Parallel implementation:** Guna processes real brochures; Lokesh uses sample chunks; Dhruvi uses a mock retriever; Shravya uses mock API responses.
3. **First integration:** Connect the complete pipeline for one company and replace all mocks in the demo flow.
4. **Full ingestion:** Process and index every company's brochures.
5. **Final verification:** Reserve the last 25–30% of available time for integration and fixes. Dhruvi coordinates backend integration; each teammate fixes their own component.

```text
PDFs → Guna: Tesseract + chunks → Lokesh: embeddings + Chroma
                                                        ↑
Shravya: UI → Dhruvi: API → Lokesh: company-filtered retrieval
                     ↓
             LLM answer + citations → UI
```

## Demo acceptance checks

| Check | Expected result | Lead |
|---|---|---|
| OCR quality | Headings, product names, and feature lists remain readable | Guna |
| Known-answer question | Correct brochure evidence supports the answer | Lokesh |
| Paraphrased question | Semantic retrieval finds evidence despite different wording | Lokesh |
| Missing information | Bot says the brochures do not provide the answer | Dhruvi |
| Company isolation | A fact available only for Company B is not used to answer for Company A | Lokesh + Dhruvi |
| Source accuracy | Cited filename and page support the answer | Dhruvi |
| Company switching | Previous company context does not contaminate the new company's answers | Shravya |
| Complete user flow | Company selection, question, answer, and citations work together | Shravya |

## Hackathon scope

Prioritize a complete, reliable pipeline with preloaded brochures and explicit company selection. Defer uploads, website research, cross-company comparisons, and other extra features until the core demo passes. The Case Study 2 PDF is reference material, not a company brochure to index.