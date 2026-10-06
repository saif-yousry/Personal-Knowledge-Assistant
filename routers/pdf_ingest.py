"""
routers/pdf_ingest.py

Endpoint to upload and ingest PDF files into the RAG pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from initializer import chunker, dispatcher, embedder, store
from loaders.pdf_loader import PDFLoader
from rag.pipeline import Pipeline
from services.app_auth_service import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["pdf-ingest"])


def _run_pdf_ingest(pdf_folder: str, cleanup: bool = True) -> None:
    """Background task: load PDFs from folder and run the RAG pipeline."""
    try:
        loader = PDFLoader(pdf_folder)
        documents = loader.load_pdfs()

        pipeline = Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
        count = pipeline.run(documents)
        logger.info("PDF ingestion complete: %d chunks stored from %s.", count, pdf_folder)
    except Exception as exc:
        logger.exception("PDF ingestion failed: %s", exc)
    finally:
        if cleanup:
            shutil.rmtree(pdf_folder, ignore_errors=True)


@router.post(
    "/ingest/pdf",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest PDF files",
    description="Uploads PDF files and runs the RAG pipeline in the background.",
)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="PDF files to ingest"),
    user=Depends(get_current_user),
):
    """Upload PDF files and ingest them into the vector store in the background."""
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No PDF files provided.")

    tmp_dir = tempfile.mkdtemp(prefix="pdf_ingest_")
    for f in pdf_files:
        path = os.path.join(tmp_dir, f.filename)
        with open(path, "wb") as out:
            content = await f.read()
            out.write(content)

    background_tasks.add_task(_run_pdf_ingest, tmp_dir, cleanup=True)
    return {
        "status": "PDF ingestion started",
        "files": [f.filename for f in pdf_files],
    }
