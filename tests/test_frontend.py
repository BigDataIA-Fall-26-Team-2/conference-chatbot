from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from streamlit.testing.v1 import AppTest
from frontend.api_client import APIError, request_json

APP = Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py'
COMPANIES = [{'company_id':'barndoor_ai','name':'Barndoor'}, {'company_id':'merge','name':'Merge'}]
ANSWER = {'answer':'Audit records go to Splunk or Datadog.', 'sources':[{'source_file':'4_Barndoor.pdf','page':2}], 'insufficient_evidence':False}


def test_chat_company_isolation_and_ten_questions():
    with patch('frontend.api_client.get_companies', return_value=COMPANIES), patch('frontend.api_client.ask_question', return_value=ANSWER) as ask:
        app = AppTest.from_file(str(APP)).run()
        assert not app.exception
        for i in range(10):
            app.chat_input[0].set_value(f'Question {i}?').run()
        assert not app.exception
        assert len(app.chat_message) == 20
        assert app.session_state['answered_total'] == 10
        assert ask.call_count == 10
        app.selectbox[0].select('merge').run()
        assert len(app.chat_message) == 0
        app.chat_input[0].set_value('Merge question?').run()
        assert ask.call_args.args[1] == 'merge'
        app.selectbox[0].select('barndoor_ai').run()
        assert len(app.chat_message) == 20
        app.button(key='clear').click().run()
        assert len(app.chat_message) == 0
        assert not app.exception


def test_errors_retry_and_missing_evidence():
    missing = {'answer':'Not stated in this brochure.', 'sources':[], 'insufficient_evidence':True}
    with patch('frontend.api_client.get_companies', return_value=COMPANIES), patch('frontend.api_client.ask_question', side_effect=[APIError('Unavailable'), missing]):
        app = AppTest.from_file(str(APP)).run()
        app.chat_input[0].set_value('Price?').run()
        assert app.error
        app.button(key='retry').click().run()
        assert len(app.chat_message) == 2
        assert app.info
        assert not app.exception


def test_backend_down_has_reconnect():
    with patch('frontend.api_client.get_companies', side_effect=APIError('Cannot connect')):
        app = AppTest.from_file(str(APP)).run()
        assert app.error
        assert app.button(key='reconnect')
        assert not app.exception


def test_timeout_is_safe_and_not_retried_automatically():
    with patch('frontend.api_client.requests.post', side_effect=requests.Timeout('private details')) as post:
        with pytest.raises(APIError, match='took too long'):
            request_json('http://localhost:8000', 'chat', {'question':'q'})
        assert post.call_count == 1
