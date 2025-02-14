from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import camelot

def load_and_convert_document(pdf_path):
    """Reads a PDF and extracts text from pages (excluding tables)."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""

def load_table_text(pdf_path):
    """Extracts table data from the PDF and returns it as text."""
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        table_texts = []
        for i, table in enumerate(tables):
            table_texts.append(f"Table {i+1}:\n" + table.df.to_csv(index=False))
        return "\n\n".join(table_texts)
    except Exception as e:
        logging.error(f"Error extracting tables from PDF: {e}")
        return ""

def get_markdown_splits(text, chunk_size=1024, chunk_overlap=300):
    """Splits text into structured chunks with increased overlap."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error splitting document: {e}")
        return []
