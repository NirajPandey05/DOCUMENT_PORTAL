
from __future__ import annotations

def extract_text_and_tables(docs: List[Document], ocr_images: bool = True) -> str:
    """
    Extracts all text and tables from a list of langchain Documents.
    If ocr_images is True, runs OCR on image documents.
    Returns a single string suitable for LLM context.
    """
    parts = []
    for d in docs:
        doc_type = d.metadata.get("type", "text")
        if doc_type == "table":
            # Table: include as CSV
            parts.append(f"\n--- TABLE ({d.metadata.get('source','')}): ---\n{d.page_content}")
        elif doc_type == "image" and ocr_images:
            # Try to OCR the image if bytes are available in metadata
            try:
                import io
                from PIL import Image
                import pytesseract
                image_bytes = d.metadata.get("image_bytes")
                if image_bytes:
                    img = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    parts.append(f"\n--- OCR IMAGE ({d.metadata.get('source','')}): ---\n{ocr_text}")
                else:
                    # Fallback: just note image present
                    parts.append(f"[IMAGE] {d.metadata.get('source','')} (no bytes for OCR)")
            except Exception as e:
                parts.append(f"[IMAGE] {d.metadata.get('source','')} (OCR failed: {e})")
        else:
            # Regular text
            parts.append(str(d.page_content))
    return "\n\n".join(parts)

from pathlib import Path
from typing import Iterable, List
from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
import pandas as pd
import pytesseract
from PIL import Image
import os
import json
from sqlalchemy import create_engine, inspect
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".ppt", ".pptx", 
    ".xlsx", ".xls", ".csv", ".md"
}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        # Accept both str and Path
        from pathlib import Path
        paths = [Path(p) if not isinstance(p, Path) else p for p in paths]
        for p in paths:
            ext = p.suffix.lower()
            # Standard document loaders
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
                docs.extend(process_pdf_with_tables(loader, p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
                docs.extend(process_docx(loader, p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
                docs.extend(process_txt(loader, p))
            # PowerPoint files
            elif ext in {".ppt", ".pptx"}:
                try:
                    from langchain_community.document_loaders import UnstructuredPowerPointLoader
                    loader = UnstructuredPowerPointLoader(str(p))
                    docs.extend(process_ppt(loader, p))
                except ImportError:
                    log.warning("Unstructured not available, skipping PowerPoint file", path=str(p))
                    continue
            # Excel and CSV files
            elif ext in {".xlsx", ".xls"}:
                try:
                    from langchain_community.document_loaders import UnstructuredExcelLoader
                    loader = UnstructuredExcelLoader(str(p), mode="elements")
                    docs.extend(process_excel(loader, p))
                except ImportError:
                    log.warning("Unstructured not available, skipping Excel file", path=str(p))
                    continue
            elif ext == ".csv":
                loader = CSVLoader(str(p))
                docs.extend(process_csv(loader, p))
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
# --- DOCX Helper ---
def process_docx(loader, docx_path: Path) -> List[Document]:
    docs = []
    try:
        docs.extend(loader.load())
    except Exception as e:
        log.error("Failed to load DOCX", path=str(docx_path), error=str(e))
    # Optionally, add image extraction for docx here
    return docs

# --- TXT Helper ---
def process_txt(loader, txt_path: Path) -> List[Document]:
    docs = []
    try:
        docs.extend(loader.load())
    except Exception as e:
        log.error("Failed to load TXT", path=str(txt_path), error=str(e))
    return docs

# --- PPT Helper ---
def process_ppt(loader, ppt_path: Path) -> List[Document]:
    docs = []
    try:
        docs.extend(loader.load())
    except Exception as e:
        log.error("Failed to load PPT", path=str(ppt_path), error=str(e))
    # Optionally, add image extraction for ppt here
    return docs

# --- Excel Helper ---
def process_excel(loader, excel_path: Path) -> List[Document]:
    docs = []
    try:
        # Load as elements (text)
        docs.extend(loader.load())
    except Exception as e:
        log.error("Failed to load Excel", path=str(excel_path), error=str(e))
    # Extract tables using pandas
    try:
        xls = pd.ExcelFile(str(excel_path))
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if not df.empty:
                docs.append(Document(
                    page_content=df.to_csv(index=False),
                    metadata={"source": str(excel_path), "type": "table", "sheet_name": sheet_name}
                ))
    except Exception as e:
        log.warning("Excel table extraction failed", path=str(excel_path), error=str(e))
    return docs

# --- CSV Helper ---
def process_csv(loader, csv_path: Path) -> List[Document]:
    docs = []
    try:
        docs.extend(loader.load())
    except Exception as e:
        log.error("Failed to load CSV", path=str(csv_path), error=str(e))
    # Extract table using pandas
    try:
        df = pd.read_csv(str(csv_path))
        if not df.empty:
            docs.append(Document(
                page_content=df.to_csv(index=False),
                metadata={"source": str(csv_path), "type": "table"}
            ))
    except Exception as e:
        log.warning("CSV table extraction failed", path=str(csv_path), error=str(e))
    return docs
# --- PDF Table & Image Extraction Helper ---
def process_pdf_with_tables(loader, pdf_path: Path) -> List[Document]:
    """
    Loads PDF as text, extracts tables (using tabula-py), and images (using fitz/PyMuPDF).
    Returns a list of langchain Documents.
    """
    docs = []
    try:
        import pdfplumber
        import pandas as pd
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "type": "text",
                            "page": page_num+1
                        }
                    ))
                    log.info("PDF page text extracted", path=str(pdf_path), page=page_num+1, text_preview=text[:200])
                # Extract tables
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if table and any(any(cell for cell in row) for row in table):
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        docs.append(Document(
                            page_content=df.to_csv(index=False),
                            metadata={
                                "source": str(pdf_path),
                                "type": "table",
                                "table_index": t_idx,
                                "page": page_num+1
                            }
                        ))
                        log.info("PDF table extracted", path=str(pdf_path), page=page_num+1, table_index=t_idx)
    except Exception as e:
        log.error("Failed to extract PDF text/tables with pdfplumber", path=str(pdf_path), error=str(e))
    # 3. Extract images using fitz (PyMuPDF)
    try:
        import fitz
        from PIL import Image
        doc = fitz.open(str(pdf_path))
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                docs.append(Document(
                    page_content=f"[IMAGE] {pdf_path.name} page {page_num+1} image {img_index+1}",
                    metadata={
                        "source": str(pdf_path),
                        "type": "image",
                        "page": page_num+1,
                        "image_index": img_index+1,
                        "image_ext": image_ext,
                        "image_bytes": image_bytes
                    }
                ))
    except Exception as e:
        log.warning("Image extraction failed or fitz not installed", path=str(pdf_path), error=str(e))
    return docs

def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def read_pdf_via_handler(handler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")