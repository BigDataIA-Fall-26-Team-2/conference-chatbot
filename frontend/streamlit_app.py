"""Run from the repository root: python -m streamlit run frontend/streamlit_app.py."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.api_client import APIError, ask_question, get_companies

load_dotenv(ROOT / '.env')
API_URL = os.getenv('API_URL', '').strip() or 'http://127.0.0.1:8000'
st.set_page_config(page_title='Conference Companion', page_icon='💬', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1060px; padding-top: 2.5rem;}
[data-testid="stSidebar"] {border-right: 1px solid #e5e9ef;}
[data-testid="stChatMessage"] {border: 1px solid #e5e9ef; border-radius: 14px; padding: 1.1rem;}
.hero-label {font-size: .75rem; font-weight: 700; letter-spacing: .16em; color: #087f72;}
h1 {letter-spacing: -.045em;}
</style>''', unsafe_allow_html=True)

st.session_state.setdefault('histories', {})
st.session_state.setdefault('answered_total', 0)

with st.sidebar:
    st.markdown('### Conference Companion')
    st.caption('Your guide to the companies behind the brochures.')
    st.divider()
    try:
        companies = get_companies(API_URL)
    except APIError as exc:
        st.error(str(exc))
        st.button('Reconnect', key='reconnect')
        st.info('Start the backend with: python -m uvicorn backend.main:app --reload')
        st.stop()
    names = {c['company_id']: c['name'] for c in companies}
    company_id = st.selectbox('Choose a company', list(names), format_func=names.get, key='company')
    st.caption('Answers use only the selected company’s brochures.')
    st.divider()
    st.metric('Questions answered this session', st.session_state.answered_total)
    st.caption('Ready for your 10-question demo. You can keep asking after ten.')
    if st.button('Clear this company’s chat', key='clear', use_container_width=True):
        st.session_state.histories[company_id] = []
        st.rerun()
    transcript = {'exported_at': datetime.now(timezone.utc).isoformat(),
                  'conversations': st.session_state.histories}
    st.download_button('Download session', json.dumps(transcript, indent=2, ensure_ascii=False),
                       file_name='conference-questions.json', mime='application/json',
                       use_container_width=True)
    st.divider()
    st.caption('Brochure-based answers • Page citations\n\nNo website research or external company facts.')

history = st.session_state.histories.setdefault(company_id, [])
st.markdown('<div class="hero-label">CONFERENCE KNOWLEDGE DESK</div>', unsafe_allow_html=True)
st.title(f'Ask about {names[company_id]}')
st.write('Go beyond the talking points. Ask about features, integrations, security, or deployment.')
st.caption('Each question is answered independently. Include the topic in follow-up questions.')


def show_message(message, index):
    with st.chat_message(message['role']):
        if message.get('error'):
            st.error(message['content'])
            return
        if message.get('company_id'):
            st.caption('Answering about ' + names.get(message['company_id'], message['company_id']))
        if message.get('insufficient_evidence'):
            st.info('The brochure does not provide enough evidence for this question.')
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Brochure sources', expanded=True):
                for source_index, source in enumerate(message['sources']):
                    st.caption(f'{source["source_file"]} · Page {source["page"]}')
                    # Only expose files actually inside this company's brochure folder.
                    directory = (ROOT / 'data/brochures' / company_id).resolve()
                    candidate = (directory / source['source_file']).resolve()
                    if candidate.parent == directory and candidate.suffix.lower() == '.pdf' and candidate.is_file():
                        st.download_button('Download brochure', candidate.read_bytes(),
                            file_name=candidate.name, mime='application/pdf',
                            key=f'pdf_{company_id}_{index}_{source_index}')


for index, message in enumerate(history):
    show_message(message, index)

suggestion = None
if not history:
    with st.container(border=True):
        st.markdown('**Start with a question**')
        columns = st.columns(3)
        for column, label, question in zip(columns,
            ['Key capabilities', 'Integrations', 'Security & controls'],
            ['What are the key capabilities described in the brochure?',
             'Which integrations and protocols does the brochure mention?',
             'What security and access controls does the brochure describe?']):
            if column.button(label, use_container_width=True):
                suggestion = question

retry_question = None
if history and history[-1].get('error'):
    if st.button('Retry last question', key='retry'):
        retry_question = history[-2]['content']
        del history[-2:]

prompt = st.chat_input(f'Ask a question about {names[company_id]}…', max_chars=4000)
question = (prompt or suggestion or retry_question or '').strip()
if question:
    history.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.spinner('Finding evidence in the brochure and preparing your answer…'):
        try:
            result = ask_question(API_URL, company_id, question)
            history.append({'role': 'assistant', 'content': result['answer'],
                            'sources': result['sources'],
                            'insufficient_evidence': result['insufficient_evidence']})
            st.session_state.answered_total += 1
        except APIError as exc:
            history.append({'role': 'assistant', 'content': str(exc), 'error': True})
    st.rerun()
