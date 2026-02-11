# Document-Based Question Answering Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions from uploaded documents using Large Language Models (LLMs) and vector search.

## Overview

This project implements a document-grounded chatbot capable of understanding and answering user queries based on uploaded documents (PDFs, text files). It leverages LangChain for document processing, vector embeddings for semantic search, and LLMs for generating context-aware answers.

## Features

- Upload and process documents in PDF or text format  
- Chunking and embedding of document content for efficient retrieval  
- Vector similarity search to retrieve relevant context for questions  
- LLM-based response generation grounded in retrieved document content  
- Interactive web interface built with Streamlit for real-time user queries  
- Modular pipeline for adding more documents or LLM models

## Tech Stack

- Python  
- LangChain  
- OpenAI / HuggingFace LLMs  
- FAISS / Chroma (vector database for embeddings)  
- Streamlit for web interface  
- Pandas, NumPy for data processing

## System Architecture

1. **Document Ingestion**  
   - Uploads and splits documents into chunks for processing.

2. **Embedding Generation**  
   - Converts document chunks into vector embeddings using LLM embedding models.

3. **Vector Search**  
   - Uses FAISS or Chroma to retrieve the most relevant document chunks for a query.

4. **Answer Generation**  
   - Generates answers using an LLM based on the retrieved context.

5. **Web Interface**  
   - Streamlit app allows users to upload documents and ask questions interactively.

## Example Usage

1. Clone the repository:git clone https://github.com/mra9165/document-qa-chatbot.git
2. Navigate to the project folder:cd document-qa-chatbot
3. Create and activate a virtual environment:
   python -m venv venv
   # Windows 
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
4. Install dependencies: pip install -r requirements.txt
5. Run the Streamlit app:streamlit run app.py




```bash
git clone https://github.com/mra9165/document-qa-chatbot.git
