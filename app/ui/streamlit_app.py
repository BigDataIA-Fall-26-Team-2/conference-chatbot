"""Conference brochure chatbot — Streamlit frontend.

Run from the repo root:
    streamlit run app/ui/streamlit_app.py

Mock mode (no backend needed): set MOCK_MODE=true in .env
Real mode: talks to the FastAPI backend at API_URL (GET /companies, POST /chat).
"""

import os
from pathlib import Path

import requests
import streamlit as st
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
TIMEOUT = 60


# ---------- data access ----------

def companies_from_yaml() -> list[dict]:
    with open(ROOT / "config" / "companies.yaml") as f:
        data = yaml.safe_load(f)["companies"]
    return [{"company_id": cid, "name": info["name"]} for cid, info in data.items()]


@st.cache_data(ttl=300)
def load_companies() -> list[dict]:
    if MOCK_MODE:
        return companies_from_yaml()
    r = requests.get(f"{API_URL}/companies", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["companies"]


def mock_chat(company_id: str, question: str) -> dict:
    # Fake response that follows the shared contract in plan.md.
    if "missing" in question.lower():
        return {
            "answer": "The brochures don't say anything about this.",
            "sources": [],
            "insufficient_evidence": True,
        }
    return {
        "answer": f"[MOCK] Placeholder answer for {company_id}: \"{question}\"",
        "sources": [{"source_file": "mock_brochure.pdf", "page": 1}],
        "insufficient_evidence": False,
    }


def ask(company_id: str, question: str) -> dict:
    if MOCK_MODE:
        return mock_chat(company_id, question)
    r = requests.post(
        f"{API_URL}/chat",
        json={"company_id": company_id, "question": question},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ---------- rendering ----------

def render_answer(resp: dict) -> None:
    if resp.get("insufficient_evidence"):
        st.warning(resp.get("answer") or "The brochures don't cover this.")
        return
    st.markdown(resp.get("answer", ""))
    sources = resp.get("sources") or []
    if sources:
        unique = list(dict.fromkeys((s["source_file"], s["page"]) for s in sources))
        with st.expander(f"Sources ({len(unique)})"):
            for file, page in unique:
                st.markdown(f"- **{file}**, page {page}")


# ---------- page ----------

st.set_page_config(page_title="Conference brochure chatbot", page_icon="📄")
st.title("Conference brochure chatbot")
st.write("Pick a company and ask about anything in its brochure.")
if MOCK_MODE:
    st.info("Mock mode is on. Answers are placeholders, not real brochure content.")

try:
    companies = load_companies()
except requests.RequestException:
    st.error(
        f"Can't reach the backend at {API_URL}. Start the API, "
        "or set MOCK_MODE=true in .env to try the UI without it."
    )
    st.stop()

names = {c["company_id"]: c["name"] for c in companies}
company_id = st.selectbox(
    "Company", options=list(names), format_func=lambda cid: names[cid]
)

# One separate chat history per company, so context never crosses companies.
if "histories" not in st.session_state:
    st.session_state.histories = {}
history = st.session_state.histories.setdefault(company_id, [])

if st.button("Clear this chat", disabled=not history):
    st.session_state.histories[company_id] = []
    st.rerun()

for turn in history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn["response"])

question = st.chat_input(f"Ask about {names[company_id]}")
if question:
    question = question.strip()
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the brochure..."):
                try:
                    resp = ask(company_id, question)
                except requests.Timeout:
                    st.error("The backend took too long to answer. Try again.")
                    st.stop()
                except requests.RequestException as e:
                    st.error(f"The backend returned an error: {e}")
                    st.stop()
            render_answer(resp)
        history.append({"question": question, "response": resp})
        st.rerun()
