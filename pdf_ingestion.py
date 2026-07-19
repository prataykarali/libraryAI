"""Shim: pdf_ingestion → archipelago.ingestion.*"""
from archipelago.ingestion.pdf_utils import *  # noqa: F403
from archipelago.ingestion.pdf_chunk import chunk_pdf
from archipelago.ingestion.pdf_formats import chunk_markdown, chunk_text, chunk_docx
from archipelago.ingestion.pdf_io import ingest_document, ingest_folder
