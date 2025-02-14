import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS

VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

def create_or_load_vector_store(filename, chunks, embeddings):
    """Creates or loads a FAISS vector store from combined chunks."""
    try:
        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{filename}.faiss"
        if vector_db_path.exists():
            return FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

        if not chunks:
            raise ValueError("No text extracted from document.")

        text_chunks = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(str(vector_db_path))
        return vector_store
    except Exception as e:
        logging.error(f"Vector store creation failed: {e}")
        return None
