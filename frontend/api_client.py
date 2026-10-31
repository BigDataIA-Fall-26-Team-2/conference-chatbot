"""Server-side HTTP client. API credentials never enter the frontend."""
import requests


class APIError(RuntimeError):
    pass


def request_json(base_url, path, payload=None):
    try:
        if payload is None:
            response = requests.get(f'{base_url.rstrip("/")}/{path}', timeout=(5, 15))
        else:
            response = requests.post(f'{base_url.rstrip("/")}/{path}', json=payload, timeout=(5, 120))
    except requests.Timeout:
        raise APIError('The answer took too long. Please try again; the previous request may still be finishing.') from None
    except requests.RequestException:
        raise APIError('Cannot reach the answer service. Make sure the backend is running, then retry.') from None
    if not response.ok:
        messages = {
            401: 'The answer service requires authentication.',
            404: 'This company is unavailable. Refresh the app and select a company again.',
            422: 'Please enter a valid company and a nonempty question.',
            429: 'The service is busy or has reached its usage limit. Please try again later.',
            502: 'The answer provider could not complete the request. Check the backend API key, quota, and model settings.',
            503: 'The answer service is not configured. Check the backend environment and retrieval setup.',
        }
        raise APIError(messages.get(response.status_code, 'The answer service encountered an error. Please try again.'))
    try:
        return response.json()
    except ValueError:
        raise APIError('The answer service returned an unreadable response.') from None


def get_companies(base_url):
    data = request_json(base_url, 'companies')
    companies = data.get('companies') if isinstance(data, dict) else None
    if not isinstance(companies, list) or not companies or not all(
        isinstance(c, dict) and isinstance(c.get('company_id'), str) and c['company_id']
        and isinstance(c.get('name'), str) and c['name'] for c in companies
    ):
        raise APIError('The company list is unavailable. Check the backend and retry.')
    return companies


def ask_question(base_url, company_id, question):
    data = request_json(base_url, 'chat', {'company_id': company_id, 'question': question})
    if not isinstance(data, dict) or not isinstance(data.get('answer'), str) or not data['answer'].strip() or type(data.get('insufficient_evidence')) is not bool or not isinstance(data.get('sources'), list):
        raise APIError('The answer service returned an incomplete response. Please retry.')
    if not all(isinstance(s, dict) and isinstance(s.get('source_file'), str)
               and type(s.get('page')) is int and s['page'] > 0 for s in data['sources']):
        raise APIError('The answer service returned invalid source references.')
    return data
