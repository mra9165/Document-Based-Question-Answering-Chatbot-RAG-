import os
import warnings
warnings.filterwarnings("ignore")
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
# load environment variables form .env files
load_dotenv(".env")
working_dir=os.path.dirname(os.path.abspath((__file__)))
embedding= HuggingFaceEmbeddings()
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
def process_document_to_chroma_db(file_name):
    persist_dir = f"{working_dir}/doc_vectorstore/{file_name}"

    loader=UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # split the text into chunks for embeddings
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts=text_splitter.split_documents(documents)
    # store the document chunks in a chroma vector database
    vectordb= Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_dir
    )
    return persist_dir
def answer_question(user_question,persist_dir):
    # load the persistent Chroma vector database
    vectordb=Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
    # create a retriver for document search
    retriever=vectordb.as_retriever()
    # create a Retrieval QA chain to answer user questions using questions uisng llama-3.3-70b-versatile
    qa_chain= RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,

    )
    response= qa_chain.invoke({"query":user_question})
    answer= response["result"]
    return answer