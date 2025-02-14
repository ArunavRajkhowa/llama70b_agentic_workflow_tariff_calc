import logging
import traceback
import json
import numpy as np
from utils.api_requests import call_groq_api
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor


class GraphState(BaseModel):
    question: str
    context: List[str]
    response: str = ""
    retry_count: int = 0

def analyze_query(state: GraphState) -> GraphState:
    logging.info(f"Analyzing query with LLM: {state.question}")
    prompt = f"""Analyze the following question and summarize all the vessel information(donot skip any) given for tariff calculation, these vessel information
    is later will be used to calculate tariff rates:\n{state.question}"""
    analysis = call_groq_api(prompt)
    state.response = analysis or "Analysis failed."
    return state


# Initialize DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_documents(docs, tokenizer, encoder):
    inputs = tokenizer(docs, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = encoder(**inputs).pooler_output
    return embeddings

def retrieve_context(state: GraphState, retriever) -> GraphState:
    logging.info(f"Retrieving context for: {state.question}")
    try:
        k = 10 if state.retry_count > 1 else 5  # Dynamic k adjustment for retries
        
        # Extract documents from FAISS vector store
        all_docs = retriever.vectorstore.docstore._dict.values()  # Retrieves all stored documents
        documents_text = [doc.page_content for doc in all_docs]

        # Sparse retrieval using BM25
        bm25 = BM25Okapi([doc.split() for doc in documents_text])
        bm25_scores = bm25.get_scores(state.question.split())
        top_k_bm25 = bm25.get_top_n(state.question.split(), documents_text, n=k)
        
        # Dense retrieval using DPR
        question_inputs = question_tokenizer(state.question, return_tensors='pt')
        with torch.no_grad():
            question_embedding = question_encoder(**question_inputs).pooler_output
        
        context_embeddings = encode_documents(top_k_bm25, context_tokenizer, context_encoder)
        
        dense_scores = [cosine_similarity(question_embedding, context_embedding.unsqueeze(0))[0][0] for context_embedding in context_embeddings]
        
        combined_scores = [(bm25_score + dense_score) / 2 for bm25_score, dense_score in zip(bm25_scores, dense_scores)]
        
        sorted_docs = [doc for _, doc in sorted(zip(combined_scores, top_k_bm25), key=lambda x: x[0], reverse=True)]
        state.context = sorted_docs
    except Exception as e:
        logging.error("Context retrieval error:", exc_info=True)
        state.context = []
    return state


def refine_retrieval(state: GraphState, retriever) -> GraphState:
    logging.info("Refining context retrieval with Hybrid approach.")
    prompt = (
    "Rank these document excerpts by how well they provide numerical tariff rates for '"
    + state.question
    + "'.\n\n"
    + "\n\n".join(state.context)
    + "\n\nReturn the most relevant sections first, including any with tables, numbers, and calculations."
)

    llm_ranked_output = call_groq_api(prompt)
    
    if llm_ranked_output:
        llm_ranked_context = llm_ranked_output.strip().split('\n')
        
        embeddings_model = retriever.vectorstore.embeddings
        question_embedding = embeddings_model.embed_query(state.question)
        context_embeddings = [embeddings_model.embed_query(doc) for doc in llm_ranked_context]
        scores = [cosine_similarity([question_embedding], [ce])[0][0] for ce in context_embeddings]
        
        sorted_context = [doc for _, doc in sorted(zip(scores, llm_ranked_context), key=lambda x: x[0], reverse=True)]
        state.context = sorted_context
    else:
        state.context = []
    return state

def perform_calculations(state: GraphState) -> GraphState:
    logging.info("Performing tariff calculation using context and all.")
    context = "\n".join(state.context) if state.context else "No context available."
    prompt = (
        "**You are a tariff calculation assistant.** Extract the required numerical values, tariff rates, and formulas from the provided context to calculate the following tariffs for the vessel:\n"
        "- Light Dues\n"
        "- Port Dues\n"
        "- Vehicle Traffic Services (VTS) Dues\n"
        "- Pilotage Dues\n"
        "- Running of Vessel Lines Dues\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Vessel Information from the question:**\n{state.question}\n\n"
        "### **Instructions:**\n"
        "1. Extract values such as GT, NT, DWT, LOA, beam, days alongside, cargo quantity, and tariff rates.\n"
        "2. For each tariff, apply the exact formula from the context.\n"
        "3. Provide a step-by-step breakdown and the final calculation.\n\n"
        "### **Example Output Format:**\n\n"
        "---\n"
        "**Port:** Durban\n\n"
        "**Tariff calculation values for this specific vessel:**\n\n"
        "➡ **Cost Item: Light Dues**\n"
        "↘ **Tariff:** Light Dues are calculated based on the gross tonnage of the vessel. The rate is 117.08 ZAR per 100 tons.\n\n"
        "**Calculation:** (51,255 GT / 100) * 117.08 ZAR = 60,062.04 ZAR\n\n"
        "**Tariff Amount:** ZAR 60,062.04\n\n"
        "**Formula:** GT / 100 * Rate\n\n"
        "**Documents:** doc 1, doc 2, doc 3\n\n"
        "**Cost Item Amount:** ZAR 60,062.04 ✅\n"
        "---\n\n"
        "➡ **Cost Item: Port Dues**\n"
        "↘ **Tariff:** Based on gross tonnage, charged per 100 tons plus a daily fee.\n\n"
        "**Calculation:** (51,255 / 100) * 192.73 ZAR + (51,255 / 100) * 57.79 ZAR * 3.396 days = 199,549.22 ZAR\n\n"
        "**Tariff Amount:** ZAR 199,549.22\n\n"
        "**Formula:** (GT/100 * Base Rate) + (GT/100 * Daily Rate * Days)\n\n"
        "**Documents:** doc 9, doc 10, doc 11\n\n"
        "**Cost Item Amount:** ZAR 199,549.22 ✅\n"
        "---\n\n"
        "➡ **Cost Item: Pilotage Dues**\n"
        "↘ **Tariff:** Calculated based on gross tonnage and pilotage rate.\n\n"
        "**Calculation:** (51,255 GT / 100) * 9.72 ZAR * 2 operations = 47,189.94 ZAR\n\n"
        "**Tariff Amount:** ZAR 47,189.94\n\n"
        "**Formula:** (GT/100 * Pilotage Rate * Operations)\n\n"
        "**Documents:** doc 8, doc 9, doc 10\n\n"
        "**Cost Item Amount:** ZAR 47,189.94 ✅\n"
        "---\n\n"
        "➡ **Cost Item: VTS Dues**\n"
        "↘ **Tariff:** Charged per GT per port call at Durban at 0.65 ZAR.\n\n"
        "**Calculation:** 51,255 GT * 0.65 ZAR = 33,315.75 ZAR\n\n"
        "**Tariff Amount:** ZAR 33,315.75\n\n"
        "**Formula:** GT * Rate\n\n"
        "**Documents:** doc 6, doc 17\n\n"
        "**Cost Item Amount:** ZAR 33,315.75 ✅\n"
        "---\n\n"
        "➡ **Cost Item: Towage**\n"
        "↘ **Tariff:** Based on GT with an incremental charge above 50,000 tons.\n\n"
        "**Calculation:** Base fee 73,118.07 ZAR + (13 increments * 32.24 ZAR) = 147,074.38 ZAR\n\n"
        "**Tariff Amount:** ZAR 147,074.38\n\n"
        "**Formula:** Base Fee + (Increments * Incremental Charge)\n\n"
        "**Documents:** doc 12, doc 13\n\n"
        "**Cost Item Amount:** ZAR 147,074.38 ✅\n"
        "---\n\n"
        "Ensure all cost items are calculated with precise values from the context and all document references are listed."
    )

    calculation_response = call_groq_api(prompt)
    logging.info(f"LLM Response: {calculation_response}")

    state.response = calculation_response if calculation_response else "Error: No response from LLM."
    return state


def check_similarity(state: GraphState, retriever) -> GraphState:
    logging.info("Checking similarity locally.")
    try:
        embeddings_model = retriever.vectorstore.embeddings
        question_embedding = embeddings_model.embed_query(state.question)
        response_embedding = embeddings_model.embed_query(state.response)
        score = cosine_similarity([question_embedding], [response_embedding])[0][0]
        logging.info(f"Similarity score: {score}")
        if score < 0.75: # having threshold 0.75 is causing repeated API calls
            state.retry_count += 1
            state.response = ""
    except Exception as e:
        logging.error("Error in similarity check:", exc_info=True)
    return state

def fallback_response(state: GraphState) -> GraphState:
    state.response = "Unable to retrieve sufficient context for tariff calculation."
    return state

def build_rag_chain(retriever, question: str) -> str:
    logging.info(f"Building RAG chain for: '{question}'")
    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve_context", lambda state: retrieve_context(state, retriever))
    graph.add_node("refine_retrieval", lambda state: refine_retrieval(state, retriever))
    graph.add_node("perform_calculations", perform_calculations)
    graph.add_node("check_similarity", lambda state: check_similarity(state, retriever))
    graph.add_node("fallback_response", fallback_response)
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve_context")
    graph.add_conditional_edges("retrieve_context", lambda state: "perform_calculations" if state.context else "refine_retrieval")
    graph.add_edge("perform_calculations", "check_similarity")
    graph.add_conditional_edges("check_similarity", lambda state: END if state.response else "refine_retrieval")
    graph.add_edge("refine_retrieval", "retrieve_context")

    compiled_graph = graph.compile()
    initial_state = GraphState(question=question, context=[], response="", retry_count=0)
    result = compiled_graph.invoke(initial_state)

    # Convert AddableValuesDict to GraphState by manually extracting
    if isinstance(result, dict):  # AddableValuesDict is dict-like
        final_response = result.get('response', None)
    else:
        final_response = getattr(result, 'response', None)

    logging.info(f"Final response: {final_response}")
    return final_response or "No valid response received."

