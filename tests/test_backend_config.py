from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from backend import config, rag
from backend.main import app


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LLM_API_KEY', raising=False)


def test_missing_key_is_actionable():
    with pytest.raises(config.BackendConfigurationError, match='OPENAI_API_KEY'):
        config.get_api_key()


@pytest.mark.parametrize('name', ['OPENAI_API_KEY', 'LLM_API_KEY'])
def test_both_key_names_supported(monkeypatch, name):
    monkeypatch.setenv(name, ' fake-test-key ')
    assert config.get_api_key() == 'fake-test-key'


def test_standard_key_preferred_and_blank_fallback(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'standard')
    monkeypatch.setenv('LLM_API_KEY', 'legacy')
    assert config.get_api_key() == 'standard'
    monkeypatch.setenv('OPENAI_API_KEY', ' ')
    assert config.get_api_key() == 'legacy'


def test_blank_model_uses_default(monkeypatch):
    monkeypatch.setenv('CHAT_MODEL', ' ')
    # Execute configuration in a separate namespace to avoid changing exception identity.
    namespace = {'__file__': config.__file__}
    exec(compile(open(config.__file__).read(), config.__file__, 'exec'), namespace)
    assert namespace['CHAT_MODEL'] == 'gpt-4o-mini'


def evidence(*args):
    return [{'id': 'test', 'company_id': 'merge', 'text': 'Connect through SCIM.',
             'source_file': '1_Merge.pdf', 'page': 1}]


def test_missing_key_endpoint_is_503(monkeypatch):
    monkeypatch.setattr(rag, 'get_chunks', evidence)
    response = TestClient(app).post('/chat', json={'company_id':'merge','question':'How?'})
    assert response.status_code == 503
    assert 'OPENAI_API_KEY' in response.json()['detail']


def test_legacy_key_reaches_sdk_without_network(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'fake-test-key')
    monkeypatch.setattr(rag, 'get_chunks', evidence)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"answer":"Through SCIM [1].","used_refs":[1],"insufficient_evidence":false}'))])
    constructor = Mock(return_value=client)
    monkeypatch.setattr(rag, 'OpenAI', constructor)
    response = TestClient(app).post('/chat', json={'company_id':'merge','question':'How?'})
    assert response.status_code == 200
    constructor.assert_called_once_with(api_key='fake-test-key')
    assert response.json()['sources'] == [{'source_file':'1_Merge.pdf','page':1}]


def test_no_evidence_and_company_list_need_no_key(monkeypatch):
    monkeypatch.setattr(rag, 'get_chunks', lambda *args: [])
    client = TestClient(app)
    assert client.get('/companies').status_code == 200
    result = client.post('/chat', json={'company_id':'merge','question':'How?'})
    assert result.status_code == 200
    assert result.json()['insufficient_evidence'] is True
