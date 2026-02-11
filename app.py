import os
import streamlit as st
from rag_utility import process_document_to_chroma_db,answer_question
working_dir = os.getcwd()
st.title(" 𔑸 Llama-3.3-70B-Document RAG")
uploaded_files=st.file_uploader("Upload a PDF file",type=["pdf"])
if uploaded_files is not None:
    # define save path
    save_path=os.path.join(working_dir,uploaded_files.name)
    # save the file
    with open(save_path,"wb") as f:
        f.write(uploaded_files.getbuffer())
    process_document=process_document_to_chroma_db(uploaded_files.name)
    st.info("Document Processed Sucessfuly")
user_question = st.text_area("Ask your question about the document")
if st.button("Answer"):
    answer = answer_question(user_question)
    st.markdown("# Llama-3.3-70B Response")
    st.markdown(answer)
    
    
