from fastapi import FastAPI, HTTPException
from .schemas import ChatRequest, ChatResponse, CompaniesResponse
from .registry import COMPANIES
from .rag import answer_question

app = FastAPI(title='Conference Brochure RAG API')

@app.get('/companies', response_model=CompaniesResponse)
def companies():
    return {'companies': [{'company_id': key, 'name': name} for key, name in COMPANIES.items()]}

@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.company_id not in COMPANIES:
        raise HTTPException(status_code=404, detail='Unknown company_id')
    try:
        return answer_question(request.company_id, request.question)
    except (ImportError, ModuleNotFoundError):
        raise HTTPException(status_code=503, detail='Retrieval service is not configured')
    except Exception:
        # Avoid leaking API credentials or other internal details.
        raise HTTPException(status_code=502, detail='Unable to generate an answer. Check backend configuration and logs.')
