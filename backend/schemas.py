from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    company_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    @field_validator('question', 'company_id')
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Must not be blank')
        return value

class Source(BaseModel):
    source_file: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    insufficient_evidence: bool

class Company(BaseModel):
    company_id: str
    name: str

class CompaniesResponse(BaseModel):
    companies: list[Company]
