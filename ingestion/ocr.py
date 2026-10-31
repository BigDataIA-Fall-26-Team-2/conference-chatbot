import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np
import pytesseract
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
BROCHURES = ROOT / "data" / "brochures"
PROCESSED = ROOT / "data" / "processed"

DPI = 300
MIN_CONF = 70
MIN_WORDS = 100
LOW_WORD_CONF = 60
MAX_LOW_SHARE = 0.12
TESS_CONFIG = "--oem 1 --psm 3"
FALLBACK = os.getenv("OCR_FALLBACK", "textract")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SKIP_LAYOUT = {"LAYOUT_PAGE_NUMBER", "LAYOUT_LIST"}

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")


def render_page(page):
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def find_skew(gray):
    small = cv2.resize(gray, None, fx=0.25, fy=0.25)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    center = (w // 2, h // 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-5, 5.25, 0.25):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = np.var(rot.sum(axis=1))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(gray):
    angle = find_skew(gray)
    if angle == 0:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    fixed = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return fixed, angle


def tesseract_page(gray):
    data = pytesseract.image_to_data(gray, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
    blocks = {}
    confs = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        confs.append(conf)
        bkey = data["block_num"][i]
        lkey = (data["par_num"][i], data["line_num"][i])
        block = blocks.setdefault(bkey, {"lines": {}, "heights": [], "top": data["top"][i], "left": data["left"][i]})
        block["lines"].setdefault(lkey, []).append(word)
        block["heights"].append(data["height"][i])
        block["top"] = min(block["top"], data["top"][i])
        block["left"] = min(block["left"], data["left"][i])

    out = []
    for bnum in sorted(blocks):
        b = blocks[bnum]
        lines = [" ".join(b["lines"][k]) for k in sorted(b["lines"])]
        out.append({
            "block": bnum,
            "text": "\n".join(lines),
            "avg_height": round(float(np.mean(b["heights"])), 1),
            "top": b["top"],
            "left": b["left"],
        })
    avg_conf = round(float(np.mean(confs)), 1) if confs else 0.0
    low_share = round(sum(1 for c in confs if c < LOW_WORD_CONF) / len(confs), 3) if confs else 1.0
    return out, avg_conf, len(confs), low_share


def textract_page(gray):
    import boto3

    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    client = boto3.client("textract", region_name=AWS_REGION)
    resp = client.analyze_document(Document={"Bytes": buf.tobytes()}, FeatureTypes=["LAYOUT"])

    by_id = {b["Id"]: b for b in resp["Blocks"]}
    confs = [b["Confidence"] for b in resp["Blocks"] if b["BlockType"] == "WORD"]
    h, w = gray.shape

    out = []
    for b in resp["Blocks"]:
        if not b["BlockType"].startswith("LAYOUT_") or b["BlockType"] in SKIP_LAYOUT:
            continue
        child_ids = [i for r in b.get("Relationships", []) if r["Type"] == "CHILD" for i in r["Ids"]]
        lines = [by_id[i] for i in child_ids if i in by_id and by_id[i]["BlockType"] == "LINE"]
        if not lines:
            continue
        box = b["Geometry"]["BoundingBox"]
        out.append({
            "block": len(out) + 1,
            "text": "\n".join(ln["Text"] for ln in lines),
            "avg_height": round(float(np.mean([ln["Geometry"]["BoundingBox"]["Height"] for ln in lines])) * h, 1),
            "top": int(box["Top"] * h),
            "left": int(box["Left"] * w),
            "layout": b["BlockType"],
        })
    avg_conf = round(float(np.mean(confs)), 1) if confs else 0.0
    return out, avg_conf, len(confs)


def docling_page(pdf_path, page_index):
    from docling.document_converter import DocumentConverter

    src = fitz.open(pdf_path)
    one = fitz.open()
    one.insert_pdf(src, from_page=page_index, to_page=page_index)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdf = Path(tmp) / "page.pdf"
        one.save(tmp_pdf)
        result = DocumentConverter().convert(str(tmp_pdf))
        md = result.document.export_to_markdown()
    blocks = [{"block": i + 1, "text": t.strip(), "avg_height": None, "top": None, "left": None}
              for i, t in enumerate(md.split("\n\n")) if t.strip()]
    return blocks, len(md.split())


def ocr_pdf(company_id, pdf_path, force_engine=None):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        gray, angle = deskew(render_page(page))
        blocks, conf, words, low_share = tesseract_page(gray)
        engine = "tesseract"
        weak = conf < MIN_CONF or words < MIN_WORDS or low_share > MAX_LOW_SHARE
        if force_engine == "textract" or (force_engine is None and weak):
            if FALLBACK == "textract" or force_engine == "textract":
                blocks, conf, words = textract_page(gray)
                engine = "textract"
            else:
                blocks, words = docling_page(pdf_path, i)
                engine = "docling"
        pages.append({
            "company_id": company_id,
            "source_file": pdf_path.name,
            "page": i + 1,
            "ocr_engine": engine,
            "ocr_confidence": conf if engine != "docling" else None,
            "word_count": words,
            "tesseract_low_share": low_share,
            "skew_degrees": angle,
            "blocks": blocks,
        })
        print(f"{company_id}\t{pdf_path.name}\tpage={i + 1}\tengine={engine}\tconf={conf}\twords={words}\tlow_share={low_share}\tskew={angle}")
    return pages


def write_outputs(company_id, pdf_path, pages):
    out_dir = PROCESSED / company_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    with open(out_dir / f"{stem}.pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out_dir / f"{stem}.md", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"<!-- page {p['page']} | {p['ocr_engine']} | conf {p['ocr_confidence']} -->\n\n")
            for b in p["blocks"]:
                f.write(b["text"] + "\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default=None)
    parser.add_argument("--engine", choices=["textract"], default=None)
    args = parser.parse_args()

    companies = [args.company] if args.company else sorted(d.name for d in BROCHURES.iterdir() if d.is_dir())
    for company_id in companies:
        for pdf_path in sorted((BROCHURES / company_id).glob("*.pdf")):
            pages = ocr_pdf(company_id, pdf_path, args.engine)
            write_outputs(company_id, pdf_path, pages)


if __name__ == "__main__":
    main()