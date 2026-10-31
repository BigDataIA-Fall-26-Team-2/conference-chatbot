"""Index chunk JSONL: python -m retrieval.index [--input PATH]."""
import argparse
import json
from pathlib import Path

from retrieval.store import ROOT, get_store


def index_chunks(chunks: list[dict]) -> None:
    """Upsert chunks by stable ID; omitted chunks are not deleted."""
    get_store().index_chunks(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/chunks.jsonl')
    args = parser.parse_args()
    with args.input.open(encoding='utf-8') as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    index_chunks(chunks)
    print(json.dumps({'indexed_chunks': len(chunks),
                      'companies': sorted({r['company_id'] for r in chunks})}, indent=2))


if __name__ == '__main__':
    main()
