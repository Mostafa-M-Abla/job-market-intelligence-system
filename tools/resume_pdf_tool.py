"""
tools/resume_pdf_tool.py — Extract readable text from a resume PDF.

WHY BYTES INSTEAD OF A FILE PATH?
----------------------------------
The original v1 tool read from a hardcoded "Resume.pdf" file on disk.
This v2 tool accepts raw PDF bytes instead, for two reasons:

  1. The FastAPI backend (Phase 2) will receive uploads via HTTP and store
     them in the database as BLOBs.  There is no guarantee of a persistent
     filesystem, especially when deployed in a container on fly.io.

  2. It avoids any dependency on a specific file path or working directory,
     making the tool easier to test in isolation.

The bytes come from the session store (Phase 1) or the `sessions` DB table
(Phase 2) — the resume_parser graph node fetches them and passes them here.

LIBRARY CHOICE
--------------
We use `pypdf` (the actively maintained successor to the deprecated PyPDF2).
pypdf extracts text from each page of the PDF and concatenates them.

LIMITATION
----------
Text extraction only works for text-based PDFs.  Scanned PDFs (where pages
are images) will produce no text.  OCR would be needed for those, but that
is out of scope for this project.
"""

import asyncio
import io

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ResumePDFInput(BaseModel):
    """Input schema for ResumePDFExtractorTool."""
    pdf_bytes: bytes = Field(description="Raw binary content of the resume PDF file")


class ResumePDFExtractorTool(BaseTool):
    """
    Extract all readable text from a resume PDF provided as raw bytes.

    Iterates over every page, extracts text, and concatenates the results
    into a single string separated by newlines.

    Raises ValueError if the PDF contains no extractable text (e.g. it is
    a scanned image PDF).
    """

    name: str = "resume_pdf_extractor"
    description: str = "Extracts text from a resume PDF provided as raw bytes."
    args_schema: type[BaseModel] = ResumePDFInput

    def _run(self, pdf_bytes: bytes) -> str:
        """
        Extract text from the PDF bytes.

        Args:
            pdf_bytes: The raw binary content of a PDF file.

        Returns:
            All text found in the PDF as a single string.

        Raises:
            ImportError:  If pypdf is not installed.
            ValueError:   If no text could be extracted (scanned/image PDF).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required: pip install pypdf")

        # Wrap the bytes in a BytesIO buffer so pypdf can read them as if
        # they were a file — no temp file needed on disk.
        reader = PdfReader(io.BytesIO(pdf_bytes))

        # Extract text from each page, using an empty string as fallback for
        # pages that return None (e.g. image-only pages).
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()

        if not text:
            raise ValueError(
                "No text extracted from PDF. The file may be a scanned image "
                "(OCR would be required to read it)."
            )
        return text

    async def _arun(self, pdf_bytes: bytes) -> str:
        """
        Async wrapper for _run().

        Runs the synchronous pypdf extraction in a thread pool so it does not
        block the async event loop used by the FastAPI server in Phase 2.
        """
        return await asyncio.to_thread(self._run, pdf_bytes)
