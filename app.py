import os
import logging
import streamlit as st
from pathlib import Path
from pdf2image import convert_from_path, exceptions
from PIL import Image
from utils.document_processing import load_and_convert_document, get_markdown_splits, load_table_text
from utils.vector_store import create_or_load_vector_store
from models.llm_groq import build_rag_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

def get_poppler_path():
    default_path = r"C:\poppler\poppler-24.08.0\Library\bin"
    return default_path if os.path.exists(default_path) else None

POPPLER_PATH = get_poppler_path()
if not POPPLER_PATH:
    st.sidebar.error("Poppler not found. Please install Poppler and add it to PATH.")

def display_pdf_in_sidebar(pdf_path, file_name):
    """Converts PDF pages to images and displays them in the sidebar."""
    try:
        images_folder = Path(VECTOR_DB_FOLDER) / file_name / "images"
        os.makedirs(images_folder, exist_ok=True)
        image_paths = sorted(images_folder.glob("*.png"))
        if image_paths:
            for i, img_path in enumerate(image_paths):
                image = Image.open(img_path)
                st.sidebar.image(image, caption=f"Page {i + 1}")
        elif POPPLER_PATH:
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            for i, image in enumerate(images):
                img_path = images_folder / f"page_{i + 1}.png"
                image.save(img_path, "PNG")
                st.sidebar.image(image, caption=f"Page {i + 1}")
        else:
            st.sidebar.warning("PDF preview is disabled due to missing Poppler.")
    except exceptions.PDFPageCountError:
        st.sidebar.error("Error: Corrupt or empty PDF.")
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        st.sidebar.error(f"Error loading PDF: {str(e)}")

st.title("Agentic RAG for Tariff Calculation")

vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")] + ["Upload New Document"]
selected_vector_db = st.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0)

if selected_vector_db == "Upload New Document":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        try:
            pdf_name = uploaded_file.name.split('.')[0]
            pdf_destination = Path(VECTOR_DB_FOLDER) / f"{pdf_name}.pdf"
            with open(pdf_destination, "wb") as f:
                f.write(uploaded_file.read())
            display_pdf_in_sidebar(pdf_destination, pdf_name)
        except Exception as e:
            logging.error(f"Error saving uploaded file: {e}")
            st.error("Failed to save the uploaded file.")

        if st.button("Process PDF and Store in Vector DB"):
            try:
                with st.spinner("Processing document..."):
                    # Load text and table data separately
                    text_content = load_and_convert_document(pdf_destination)
                    table_content = load_table_text(pdf_destination)
                    # Split each separately
                    text_chunks = get_markdown_splits(text_content)
                    table_chunks = get_markdown_splits(table_content)
                    # Combine both chunk lists
                    combined_chunks = text_chunks + table_chunks
                    # Use a financial domain-specific embedding model
                    embeddings = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")
                    vector_store = create_or_load_vector_store(pdf_name, combined_chunks, embeddings)
                    
                    if vector_store:
                        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{pdf_name}.faiss"
                        vector_store.save_local(str(vector_db_path))
                        st.success("PDF processed and stored in the vector database.")
                    else:
                        st.error("Failed to create/load vector database. Check logs.")
            except Exception as e:
                logging.error(f"Processing error: {e}")
                st.error("An error occurred during document processing.")

elif selected_vector_db != "Upload New Document":
    try:
        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
        if vector_db_path.exists():
            # Use a financial domain-specific embedding model
            embeddings = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")
            vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)
            pdf_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.pdf"
            if pdf_path.exists():
                display_pdf_in_sidebar(pdf_path, selected_vector_db)
            else:
                st.sidebar.warning("PDF file not found for the selected vector DB.")
        else:
            st.sidebar.error("Selected vector DB file does not exist.")
    except Exception as e:
        logging.error(f"Error loading vector DB: {e}")
        st.error("Failed to load vector database.")

question = st.text_input("Enter your question:")

if st.button("Submit Question") and question.strip():
    logging.info(f"Submit button clicked. Question: '{question}'")
    with st.spinner("Processing with Agentic AI..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")
            vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
            vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

            # Use the vector store as a retriever with MMR
            mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

            # Implementing Hybrid Search
            # bm25_retriever = BM25Retriever.from_documents(vector_store.similarity_search(question, k=5))

            # Combining retrievers for hybrid
            # hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, mmr_retriever], weights=[0.4, 0.6])

            response = build_rag_chain(mmr_retriever, question)
            if response.startswith("{") or response.startswith("["):
                st.json(response)
            else:
                st.markdown(response)

            logging.info("Question processing completed.")
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            st.error("An error occurred while processing the question.")

 
