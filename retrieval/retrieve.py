"""Query one company: python -m retrieval.retrieve COMPANY 'QUESTION'."""
import argparse
import json

from retrieval.store import get_store


def retrieve(company_id: str, question: str, k: int = 6) -> list[dict]:
    """Return evidence with cosine similarity score (higher is better).

    Scores are not probabilities or proof that the question is answerable.
    Unknown companies raise ValueError; valid unindexed companies return [].
    """
    return get_store().retrieve(company_id, question, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('company_id')
    parser.add_argument('question')
    parser.add_argument('--k', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(retrieve(args.company_id, args.question, args.k), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
