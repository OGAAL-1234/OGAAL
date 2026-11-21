import io
import os
import zipfile
from typing import Optional
import docx
import pandas as pd
import pytesseract
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pypdf import PdfReader

app = FastAPI(title="OGAAL Universal Pollinations API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PROFILES = {
    "somnia": {
        "preferred": ["openai-fast", "openai", "mistral"],
        "hint": "U jawaab si degdeg ah, kooban, oo toos ah."
    },
    "taskora": {
        "preferred": ["openai", "mistral", "openai-fast"],
        "hint": "Si nidaamsan u xalli hawsha ama su'aasha tallaabo-tallaabo."
    },
    "cereon": {
        "preferred": ["mistral", "openai", "openai-fast"],
        "hint": "U jawaab si dabiici ah oo ku habboon shaqooyinka maalmeedka."
    },
    "nexcode": {
        "preferred": ["qwen-coder", "openai", "mistral"],
        "hint": "Diiradda saar saxnaanta farsamo, koodhka, iyo xallinta dhibaatooyinka adag."
    },
}

EFFORT_CONFIG = {
    "low": {"max_tokens": 600, "max_chars": 8000},
    "medium": {"max_tokens": 1400, "max_chars": 16000},
    "high": {"max_tokens": 2600, "max_chars": 28000},
}

TEXT_EXTS = {
    "txt", "md", "csv", "json", "py", "js", "ts", "jsx", "tsx",
    "html", "css", "xml", "yaml", "yml", "log", "sql", "env", "ini", "sh"
}

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}


def extract_bytes_by_name(filename: str, data: bytes, depth: int = 0) -> str:
    if depth > 2:
        return ""
    ext = filename.rsplit(".", 1)[-1].lower().strip() if filename and "." in filename else ""

    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="ignore").strip()

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        pages_out = []
        for i, page in enumerate(reader.pages, 1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages_out.append(f"--- Bogga {i} ---\n{txt}")
        if pages_out:
            return "\n\n".join(pages_out).strip()
        return f"[Faylka PDF ({filename}) qoraal toos ah lagama helin—waxaa laga yaabaa inuu sawir yahay.]"

    if ext == "docx":
        doc = docx.Document(io.BytesIO(data))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                vals = [c.text.strip() for c in row.cells if c.text.strip()]
                if vals:
                    lines.append(" | ".join(vals))
        return "\n".join(lines).strip()

    if ext in {"xlsx", "xls"}:
        xl = pd.ExcelFile(io.BytesIO(data))
        sheets = []
        for s in xl.sheet_names:
            df = xl.parse(s)
            sheets.append(f"--- Sheet: {s} ---\n{df.to_csv(index=False)}")
        return "\n\n".join(sheets).strip()

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(io.BytesIO(data))
            ocr_text = pytesseract.image_to_string(img).strip()
            meta = f"Sawir ({filename}) | Cabbirka: {img.size[0]}x{img.size[1]} | Qaabka: {img.format}"
            if ocr_text:
                return f"{meta}\n--- Qoraalka Sawirka (OCR) ---\n{ocr_text}"
            return f"{meta}\n[Qoraal lagama helin sawirkan.]"
        except Exception as e:
            return f"[Sawirka {filename} lama akhriyi karin: {e}]"

    if ext == "zip":
        parts = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            parts.append(f"Faylasha ku jira ZIP ({len(names)}): {', '.join(names[:30])}")
            for name in names[:8]:
                try:
                    inner = zf.read(name)
                    extracted = extract_bytes_by_name(name, inner, depth + 1)
                    if extracted:
                        parts.append(f"=== Gudaha ZIP: {name} ===\n{extracted[:3000]}")
                except Exception:
                    continue
        return "\n\n".join(parts).strip()

    decoded = data.decode("utf-8", errors="ignore")
    printable = "".join(ch for ch in decoded if ch.isprintable() or ch in "\n\r\t").strip()
    if len(printable) >= 20:
        return f"[Faylka .{ext or 'unknown'} waxaa loo akhriyey qoraal guud]\n{printable}"

    raise HTTPException(
        status_code=400,
        detail=f"Faylka (.{ext or 'unknown'}) qoraal sax ah lagama soo saari karo."
    )


def get_live_models() -> list[str]:
    try:
        r = requests.get("https://text.pollinations.ai/models", timeout=15)
        r.raise_for_status()
        raw = r.json()
        out = []
        for item in raw:
            name = item if isinstance(item, str) else item.get("name") if isinstance(item, dict) else None
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


def build_payload(
    question: str,
    ogaal_model: str,
    effort: str,
    filename: Optional[str] = None,
    extracted_text: Optional[str] = None,
):
    profile = MODEL_PROFILES.get((ogaal_model or "taskora").lower().strip(), MODEL_PROFILES["taskora"])
    eff = EFFORT_CONFIG.get((effort or "medium").lower().strip(), EFFORT_CONFIG["medium"])
    q = question.strip() if question and question.strip() else "Soo koob oo sharrax."

    system_msg = (
        "Waxaad tahay caawiye ka tirsan OGAAL. Haddii xog fayl (`converted_file`) laguu soo gudbiyo, "
        "u tixgeli inay tahay xog laga soo saaray fayl rasmi ah ee la rogay, ee ma aha farriin chat "
        "caadi ah. Jawaabtaada ku dhis xogta dhabta ah ee ku jirta faylka marka fayl la soo diro. "
        + profile["hint"]
    )

    if filename and extracted_text:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        trimmed = extracted_text[:eff["max_chars"]]
        truncated = len(extracted_text) > eff["max_chars"]
        user_msg = (
            f"[SOURCE_TYPE: CONVERTED_FILE]\n"
            f"Filename: {filename}\n"
            f"Extension: .{ext}\n"
            f"Truncated: {'yes' if truncated else 'no'}\n"
            f"--- FILE CONTENT START ---\n{trimmed}\n--- FILE CONTENT END ---\n\n"
            f"Question: {q}"
        )
        return system_msg, user_msg, eff["max_tokens"], truncated, "converted_file"

    return system_msg, q, eff["max_tokens"], False, "chat_prompt"


@app.get("/")
def root():
    return {"status": "ok", "service": "OGAAL Universal Pollinations API"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "OGAAL Universal Pollinations API"}


@app.post("/analyze-file")
async def analyze_file(
    file: Optional[UploadFile] = File(None),
    question: str = Form("Soo koob faylkan."),
    model: str = Form("taskora"),
    effort: str = Form("medium"),
):
    filename = None
    extracted_text = None

    if file is not None and file.filename:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Faylku waa madhan yahay.")
        filename = file.filename
        extracted_text = extract_bytes_by_name(filename, raw)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Qoraal lagama helin faylkan.")

    sys_msg, usr_msg, max_tokens, truncated, source_type = build_payload(
        question=question,
        ogaal_model=model,
        effort=effort,
        filename=filename,
        extracted_text=extracted_text,
    )

    profile = MODEL_PROFILES.get((model or "taskora").lower().strip(), MODEL_PROFILES["taskora"])
    live = get_live_models()

    candidates = [m for m in profile["preferred"] if not live or m in live]
    for m in live:
        if m not in candidates:
            candidates.append(m)
        if len(candidates) >= 6:
            break
    if not candidates:
        candidates = ["openai", "mistral"]

    last_err = None
    for poll_model in candidates:
        try:
            resp = requests.post(
                "https://text.pollinations.ai/",
                json={
                    "model": poll_model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": usr_msg},
                    ],
                },
                timeout=90,
            )
            resp.raise_for_status()
            if "application/json" in resp.headers.get("Content-Type", ""):
                data = resp.json()
                reply = (
                    data["choices"][0].get("message", {}).get("content", "")
                    if isinstance(data, dict) and data.get("choices")
                    else str(data)
                )
            else:
                reply = resp.text

            if reply and reply.strip():
                return {
                    "status": "success",
                    "source_type": source_type,
                    "filename": filename,
                    "ogaal_model": model,
                    "effort": effort,
                    "pollinations_model": poll_model,
                    "truncated": truncated,
                    "reply": reply.strip(),
                }
        except Exception as exc:
            last_err = str(exc)

    raise HTTPException(status_code=502, detail=f"Fallback-gu wuu fashilmay: {last_err}")
