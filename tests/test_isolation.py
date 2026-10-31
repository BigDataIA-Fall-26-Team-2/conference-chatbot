"""Real Chroma integration tests; deterministic embeddings avoid model downloads."""
import numpy as np
import pytest

from retrieval.store import VectorStore


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] if 'secret' in t else [0., 1.] for t in texts])


def chunk(company, text, key='1'):
    return dict(id=f'{company}_{key}', text=text, company_id=company,
                source_file=f'{company}.pdf', page=1, chunk_index=1)


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / 'db', encoder=Encoder(), companies=['alpha', 'beta', 'empty'])


def test_other_company_perfect_match_never_returned(store):
    store.index_chunks([chunk('alpha', 'ordinary'), chunk('beta', 'secret')])
    found = store.retrieve('alpha', 'secret', k=100)
    assert len(found) == 1
    assert found[0]['company_id'] == 'alpha'
    assert found[0]['text'] == 'ordinary'
    assert store.retrieve('empty', 'secret') == []


def test_persistence_upsert_and_order(store):
    records = [chunk('alpha', 'ordinary'), chunk('alpha', 'secret', '2')]
    store.index_chunks(records)
    store.index_chunks(records)
    assert store.collection('alpha').count() == 2
    reopened = VectorStore(path=store.client.get_settings().persist_directory,
                           encoder=Encoder(), companies=['alpha'])
    found = reopened.retrieve('alpha', 'secret')
    assert found[0]['id'] == 'alpha_2'
    assert found[0]['score'] > found[1]['score']
    assert found[0]['source_file'] == 'alpha.pdf'
    store.index_chunks([chunk('alpha', 'updated', '2')])
    assert store.collection('alpha').count() == 2
    assert 'updated' in [r['text'] for r in store.retrieve('alpha', 'anything')]


def test_invalid_inputs_fail_closed(store):
    for company in ('', 'unknown', None):
        with pytest.raises(ValueError):
            store.retrieve(company, 'secret')
    with pytest.raises(ValueError):
        store.retrieve('alpha', ' ')
    with pytest.raises(ValueError):
        store.retrieve('alpha', 'question', k=0)
    with pytest.raises(ValueError):
        store.index_chunks([chunk('alpha', 'valid'), chunk('unknown', 'bad')])
    assert store.collection('alpha') is None
    with pytest.raises(ValueError):
        store.index_chunks([chunk('alpha', 'one'), chunk('alpha', 'duplicate')])


def test_model_mismatch_rejected(store):
    store.index_chunks([chunk('alpha', 'ordinary')])
    other = VectorStore(path=store.client.get_settings().persist_directory,
                        model_name='different-model', encoder=Encoder(), companies=['alpha'])
    with pytest.raises(ValueError, match='Embedding model changed'):
        other.retrieve('alpha', 'question')


def test_metadata_filter_excludes_contaminated_record(store):
    store.index_chunks([chunk('alpha', 'ordinary')])
    store.collection('alpha').upsert(ids=['bad'], documents=['secret'],
        embeddings=[[1., 0.]], metadatas=[{'company_id': 'beta'}])
    assert [r['id'] for r in store.retrieve('alpha', 'secret')] == ['alpha_1']


@pytest.mark.parametrize('field,value', [
    ('text', ''), ('text', None), ('id', ' '), ('source_file', ''),
    ('page', 0), ('page', True), ('page', '1'), ('chunk_index', -1),
])
def test_bad_batch_does_not_write_valid_prefix(store, field, value):
    bad = chunk('alpha', 'bad', '2')
    bad[field] = value
    with pytest.raises(ValueError):
        store.index_chunks([chunk('alpha', 'valid'), bad])
    assert store.collection('alpha') is None


@pytest.mark.parametrize('question', [None, '', '\t', 12, []])
def test_invalid_questions(store, question):
    with pytest.raises(ValueError):
        store.retrieve('alpha', question)


@pytest.mark.parametrize('k', [True, -1, 0, 1.5, '6', None])
def test_invalid_limits(store, k):
    with pytest.raises(ValueError):
        store.retrieve('alpha', 'question', k)


def test_empty_batch_noop(store):
    store.index_chunks([])
    assert store.client.list_collections() == []


def test_large_batch_and_json_contract(store):
    import json
    store.index_chunks([chunk('alpha', f'text {i}', str(i)) for i in range(130)])
    assert store.collection('alpha').count() == 130
    results = store.retrieve('alpha', 'text', k=7)
    assert len(results) == 7
    assert all(set(r) == {'id', 'text', 'company_id', 'source_file', 'page', 'chunk_index', 'score'} for r in results)
    json.dumps(results, allow_nan=False)


def test_collection_tampering_rejected(store):
    store.index_chunks([chunk('alpha', 'ordinary')])
    store.collection('alpha').modify(metadata={'company_id': 'beta', 'embedding_model': store.model_name})
    with pytest.raises(RuntimeError, match='company metadata mismatch'):
        store.retrieve('alpha', 'secret')


def test_post_query_guard_rejects_wrong_company(store, monkeypatch):
    class CorruptCollection:
        def count(self):
            return 1
        def query(self, **kwargs):
            return dict(ids=[['bad']], documents=[['secret']],
                        metadatas=[[{'company_id': 'beta'}]], distances=[[0.]])
    monkeypatch.setattr(store, 'collection', lambda *a, **kw: CorruptCollection())
    with pytest.raises(RuntimeError, match='isolation check failed'):
        store.retrieve('alpha', 'secret')


def test_public_backend_functions(store, monkeypatch):
    from retrieval import index as index_module, retrieve as retrieve_module
    monkeypatch.setattr(index_module, 'get_store', lambda: store)
    monkeypatch.setattr(retrieve_module, 'get_store', lambda: store)
    assert index_module.index_chunks([chunk('alpha', 'secret')]) is None
    assert retrieve_module.retrieve('alpha', 'secret', 1)[0]['text'] == 'secret'


def test_threaded_company_requests(store):
    from concurrent.futures import ThreadPoolExecutor
    store.index_chunks([chunk('alpha', 'ordinary'), chunk('beta', 'secret')])
    companies = ['alpha', 'beta'] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda c: store.retrieve(c, 'secret'), companies))
    assert all(all(r['company_id'] == c for r in rows) for c, rows in zip(companies, results))


def test_non_mapping_chunk_has_clear_error(store):
    with pytest.raises(ValueError, match='record'):
        store.index_chunks([None])


def test_persistence_in_separate_process(store):
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from retrieval.store import ROOT
    store.index_chunks([chunk('alpha', 'secret')])
    script = '''
import json, sys, numpy as np
from retrieval.store import VectorStore
class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] for _ in texts])
store = VectorStore(path=sys.argv[1], encoder=Encoder(), companies=['alpha'])
print(json.dumps(store.retrieve('alpha', 'secret')))
'''
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    result = subprocess.run([sys.executable, '-c', script, store.client.get_settings().persist_directory],
                            cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout)[0]['id'] == 'alpha_1'


def test_embedding_failure_can_be_retried(store):
    class FailedEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError('Embedding unavailable')
    store._encoder = FailedEncoder()
    with pytest.raises(RuntimeError, match='Embedding unavailable'):
        store.index_chunks([chunk('alpha', 'secret')])
    assert store.collection('alpha').count() == 0
    store._encoder = Encoder()
    store.index_chunks([chunk('alpha', 'secret')])
    assert store.retrieve('alpha', 'secret')[0]['text'] == 'secret'
