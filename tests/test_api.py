import io

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
TEXT = "TrueNorth learns with on-chip plasticity during deployment. SNNs always beat ANNs."


def test_health_and_ui():
    assert client.get("/api/health").json() == {"status": "healthy"}
    assert "<html" in client.get("/").text.lower()


def test_review_text():
    r = client.post("/api/review-text", json={"text": TEXT})
    assert r.status_code == 200
    assert r.json()["result"]["final_report"]["claims_flagged"] == 2


@pytest.mark.parametrize("payload,status", [
    ({"text": "short"}, 400),          # original: HTTP 200 with an {"error": ...} body the UI could not render
    ({"text": 123}, 422),              # original: HTTP 500 (AttributeError on int.strip)
    ({}, 422),
    ({"text": None}, 422),
])
def test_review_text_rejects_bad_payloads(payload, status):
    assert client.post("/api/review-text", json=payload).status_code == status


def test_upload_txt():
    r = client.post("/api/review", files={"file": ("doc.txt", TEXT.encode(), "text/plain")})
    assert r.status_code == 200
    assert r.json()["filename"] == "doc.txt"
    assert r.json()["result"]["final_report"]["issues_found"] >= 2


def test_upload_docx_roundtrip():
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph(TEXT)
    doc.save(buf)
    r = client.post("/api/review", files={"file": ("doc.docx", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200
    assert r.json()["result"]["final_report"]["claims_flagged"] == 2


def test_corrupt_pdf_is_an_error_not_a_reviewed_document():
    """The original returned '[PDF extraction error: ...]' as text and then reviewed that string."""
    r = client.post("/api/review", files={"file": ("x.pdf", b"this is not a pdf at all, just bytes", "application/pdf")})
    assert r.status_code == 422


def test_unsupported_type_and_oversize_rejected(monkeypatch):
    assert client.post("/api/review", files={"file": ("x.doc", b"abc", "application/msword")}).status_code == 400
    assert client.post("/api/review", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 400
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/api/review", files={"file": ("x.txt", b"x" * 50, "text/plain")}).status_code == 413


def test_uploads_are_not_written_to_disk():
    client.post("/api/review", files={"file": ("doc.txt", TEXT.encode(), "text/plain")})
    assert not (main.STATIC_DIR.parent / "uploads").exists()
