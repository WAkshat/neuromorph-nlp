"""Extract text from uploaded documents (txt, pdf, docx), in memory."""

import io
from pathlib import Path

SUPPORTED_SUFFIXES = (".txt", ".pdf", ".docx")


class ExtractionError(ValueError):
    """The file could not be read as its declared type."""


def extract_text(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join(t for page in reader.pages if (t := page.extract_text()))
        if suffix == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:  # pypdf / python-docx raise many unrelated exception types
        raise ExtractionError(f"Could not read {suffix} file: {e}") from e
    raise ExtractionError(f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_SUFFIXES)}")
