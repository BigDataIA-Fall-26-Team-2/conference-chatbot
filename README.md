# Brochure RAG chatbot

Chatbot that answers questions only from conference brochures. Each company is kept in its own Chroma collection so no company data leaks into another company's answer.

Pipeline: PDF -> OCR (Tesseract, Docling fallback) -> chunks with company metadata -> Chroma -> FastAPI -> Streamlit

## Setup
pip install -r requirements.txt
copy .env.example .env   (then fill the keys)

## Add a new brochure
Put the PDF in data/brochures/<company_id>/, add the company to config/companies.yaml, run ingest.

## Run
python ingest/ocr.py
python ingest/chunk.py
python ingest/index.py
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
