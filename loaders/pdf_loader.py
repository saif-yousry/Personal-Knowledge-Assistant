"""
Load PDF documents from a specified folder.
"""

import os
from typing import List

import pymupdf

from models import Document


class PDFLoader:
    """
    Load PDF documents from a specified folder.
    Each PDF is split into pages, and each page is represented as a `Document`
    """
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder

    def load_pdfs(self) -> List[Document]:
        """
        Load PDF documents from the specified folder.

        Returns
        -------
        List[Document]
            A list of Document objects representing the loaded PDF pages.
        """
        documents = []

        for filename in os.listdir(self.pdf_folder):

            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.pdf_folder, filename)
            # PyMuPDF: read, modify, create, and extract data from PDFs
            with pymupdf.open(pdf_path) as pdf:

                for page_num in range(len(pdf)):

                    page = pdf.load_page(page_num)

                    text = page.get_text()

                    if text.strip():

                        documents.append(Document(
                            source=filename,
                            page=page_num + 1,
                            doc_type="pdf",
                            text=text.strip(),
                        ))

        return documents
