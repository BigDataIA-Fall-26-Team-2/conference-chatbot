"""Persistent, company-scoped vector storage shared by indexing and retrieval."""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import yaml
from chromadb.errors import NotFoundError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '


class VectorStore:
    def __init__(self, path=None, model_name=None, encoder=None, companies=None):
        self.model_name = model_name or os.getenv('EMBED_MODEL') or DEFAULT_MODEL
        location = Path(path or os.getenv('CHROMA_DIR') or 'chroma_db')
        self.client = chromadb.PersistentClient(path=str(location if location.is_absolute() else ROOT / location))
        if companies is None:
            companies = yaml.safe_load((ROOT / 'config/companies.yaml').read_text())['companies']
        self.companies = set(companies)
        self._encoder = encoder

    def validate_company(self, company_id):
        if not isinstance(company_id, str) or company_id not in self.companies:
            raise ValueError(f'Unknown company_id: {company_id!r}')

    def collection(self, company_id, create=False):
        self.validate_company(company_id)
        name = f'brochures_{company_id}'
        try:
            collection = self.client.get_collection(name, embedding_function=None)
        except NotFoundError:
            if not create:
                return None
            collection = self.client.create_collection(
                name, embedding_function=None,
                metadata={'company_id': company_id, 'embedding_model': self.model_name},
                configuration={'hnsw': {'space': 'cosine'}},
            )
        metadata = collection.metadata or {}
        if metadata.get('company_id') != company_id:
            raise RuntimeError('Collection company metadata mismatch')
        if metadata.get('embedding_model') != self.model_name:
            raise ValueError('Embedding model changed; use a fresh CHROMA_DIR and reindex all chunks')
        return collection

    def embed(self, texts, query=False):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        if query and self.model_name == DEFAULT_MODEL:
            texts = [QUERY_PREFIX + text for text in texts]
        return self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def index_chunks(self, chunks):
        # Validate the complete batch before writing anything.
        groups, seen = {}, set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Each chunk record must be a dictionary")
            for key in ('id', 'text', 'company_id', 'source_file'):
                if not isinstance(chunk.get(key), str) or not chunk[key].strip():
                    raise ValueError(f'Chunk requires nonempty {key}')
            self.validate_company(chunk['company_id'])
            for key in ('page', 'chunk_index'):
                if type(chunk.get(key)) is not int or chunk[key] < 1:
                    raise ValueError(f'Chunk requires positive integer {key}')
            if chunk['id'] in seen:
                raise ValueError(f'Duplicate chunk ID: {chunk["id"]}')
            seen.add(chunk['id'])
            groups.setdefault(chunk['company_id'], []).append(chunk)
        for company_id in groups:
            self.collection(company_id)  # Check existing model metadata before writing.
        for company_id, records in groups.items():
            collection = self.collection(company_id, create=True)
            for start in range(0, len(records), 64):
                batch = records[start:start + 64]
                collection.upsert(
                    ids=[r['id'] for r in batch],
                    documents=[r['text'] for r in batch],
                    embeddings=self.embed([r['text'] for r in batch]),
                    metadatas=[{k: r[k] for k in ('company_id', 'source_file', 'page', 'chunk_index')} for r in batch],
                )

    def retrieve(self, company_id, question, k=6):
        self.validate_company(company_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be nonempty')
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer')
        collection = self.collection(company_id)
        if collection is None or collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=self.embed([question.strip()], query=True),
            where={'company_id': company_id},
            n_results=min(k, collection.count()),
            include=['documents', 'metadatas', 'distances'],
        )
        output = []
        for chunk_id, text, metadata, distance in zip(
            results['ids'][0], results['documents'][0],
            results['metadatas'][0], results['distances'][0],
        ):
            if metadata.get('company_id') != company_id:
                raise RuntimeError('Company isolation check failed')
            output.append({'id': chunk_id, 'text': text, **metadata,
                           'score': 1.0 - float(distance)})
        return output


@lru_cache(maxsize=1)
def get_store():
    return VectorStore()
