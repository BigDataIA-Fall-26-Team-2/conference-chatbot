import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BROCHURES = ROOT / "data" / "brochures"
PROCESSED = ROOT / "data" / "processed"
OUT_FILE = PROCESSED / "chunks.jsonl"

MAX_WORDS = 150
OVERLAP_MAX_WORDS = 40
MIN_REAL_RATIO = 0.5
MIN_WORDS = 3

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9.&+'/-]*")

OCR_FIXES = [
    (re.compile(r"\bAl\b"), "AI"),
    (re.compile(r"\bPil\b"), "PII"),
    (re.compile(r"\bIl\b"), "II"),
]


def clean_text(text):
    text = text.replace("\u2014", ", ").replace("\u2013", "-")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    for pattern, fix in OCR_FIXES:
        text = pattern.sub(fix, text)
    return text


def is_noise(text):
    tokens = text.split()
    if len(tokens) < MIN_WORDS:
        return True
    real = sum(1 for t in tokens if WORD_RE.fullmatch(t.strip(",.:;()\"!?")) and len(t.strip(",.:;()\"!?")) >= 2)
    return real / len(tokens) < MIN_REAL_RATIO


def load_pages(company_id, pdf_path):
    pages_file = PROCESSED / company_id / f"{pdf_path.stem}.pages.jsonl"
    if not pages_file.exists():
        from ingestion.ocr import ocr_pdf, write_outputs
        write_outputs(company_id, pdf_path, ocr_pdf(company_id, pdf_path))
    with open(pages_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def page_chunks(blocks):
    chunks, current, count = [], [], 0
    for b in blocks:
        words = len(b.split())
        if current and count + words > MAX_WORDS:
            chunks.append(current)
            last = current[-1]
            current = [last] if len(last.split()) <= OVERLAP_MAX_WORDS else []
            count = sum(len(x.split()) for x in current)
        current.append(b)
        count += words
    if current:
        chunks.append(current)
    return ["\n\n".join(c) for c in chunks]


def chunk_pdf(company_id, pdf_path):
    records = []
    chunk_index = 0
    for page in load_pages(company_id, pdf_path):
        blocks = [clean_text(b["text"]) for b in page["blocks"]]
        blocks = [b for b in blocks if b and not is_noise(b)]
        for text in page_chunks(blocks):
            chunk_index += 1
            records.append({
                "id": f"{company_id}_{pdf_path.stem}_p{page['page']}_c{chunk_index}",
                "text": text,
                "company_id": company_id,
                "source_file": pdf_path.name,
                "page": page["page"],
                "chunk_index": chunk_index,
            })
    return records


def ingest_folder(path: str) -> list[dict]:
    base = Path(path)
    records = []
    for company_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        for pdf_path in sorted(company_dir.glob("*.pdf")):
            records.extend(chunk_pdf(company_dir.name, pdf_path))
    return records


def main():
    records = ingest_folder(str(BROCHURES))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records:
        print(f"{r['id']}\twords={len(r['text'].split())}")
    print(f"total_chunks={len(records)}")


if __name__ == "__main__":
    main()