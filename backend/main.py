"""Neuromorphic-claims document reviewer: FastAPI backend."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.orchestrator import DocumentTooShort, MultiAgentOrchestrator, PipelineConfig
from services.document_processor import SUPPORTED_SUFFIXES, ExtractionError, extract_text

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("reviewer")

app = FastAPI(
    title="Neuromorphic Claims Reviewer",
    description="Flags overclaims and statements that contradict a curated neuromorphic-computing knowledge base.",
    version="2.0.0",
)

# The API holds no credentials or sessions, so any origin may call it (local frontends).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# REVIEWER_NLI=1 enables the optional NLI verifier (needs torch + transformers, downloads a model).
orchestrator = MultiAgentOrchestrator(PipelineConfig(nli=os.environ.get("REVIEWER_NLI") == "1"))


class TextPayload(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)


def _review(text: str) -> dict:
    try:
        result = orchestrator.process_document(text)
    except DocumentTooShort as e:
        raise HTTPException(status_code=400, detail=str(e))
    report = result["final_report"]
    log.info("reviewed document=%s claims=%d flagged=%d issues=%d", result["document_id"],
             report["claim_count"], report["claims_flagged"], report["issues_found"])
    return result


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "healthy"}


# Plain `def` endpoints: FastAPI runs them in a worker thread, so a long document
# does not block the event loop for other requests.
@app.post("/api/review")
def review_document(file: UploadFile = File(...)):
    """Upload a .txt / .pdf / .docx document for review. Nothing is written to disk."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(SUPPORTED_SUFFIXES)}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    try:
        text = extract_text(data, file.filename)
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    result = _review(text)
    return {"filename": file.filename, "document_id": result["document_id"], "result": result}


@app.post("/api/review-text")
def review_text(payload: TextPayload):
    """Review plain text sent as JSON: {"text": "..."}."""
    return {"result": _review(payload.text)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
