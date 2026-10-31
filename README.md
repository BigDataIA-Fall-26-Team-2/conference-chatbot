# Brochure RAG chatbot

Chatbot that answers questions only from conference brochures. Each company is kept in its own Chroma collection so no company data leaks into another company's answer.

Pipeline: PDF -> OCR (Tesseract, Docling fallback) -> chunks with company metadata -> Chroma -> FastAPI -> Streamlit

## Install
conda create -n brochure-rag python=3.11 -y
conda activate brochure-rag
pip install -r requirements.txt
Install Tesseract (Windows: UB Mannheim installer), set TESSERACT_CMD in .env

## Add a new brochure
Put the PDF in data/brochures/<company_id>/, add the company to config/companies.yaml, run ingestion.

## Run
python ingestion/ocr.py
uvicorn backend.main:app --reload
streamlit run frontend/streamlit_app.py

## Add a new brochure
Put the PDF in data/brochures/<company_id>/, add the company to config/companies.yaml, run ingest.

## Run
python ingest/ocr.py
python ingest/chunk.py
python ingest/index.py
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
