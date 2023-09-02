__import__('pysqlite3')
import sys
import pickle
import subprocess
import os
import tempfile
import streamlit as st

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Streamlit UI
st.title("ChatPDF")
st.write("---")
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def git_clone_and_commit_and_push(db):
    try:
        subprocess.run(['git', 'clone', 'https://github.com/sssc2023/prompt_day.git'], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(['git', '-C', 'prompt_day', 'pull'], check=True)

    db_path = 'prompt_day/db.pkl'
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)

    try:
        subprocess.run(['git', '-C', 'prompt_day', 'add', 'db.pkl'], check=True)
        subprocess.run(['git', '-C', 'prompt_day', 'commit', '-m', 'Add new db object'], check=True)
        subprocess.run(['git', '-C', 'prompt_day', 'push', 'origin', 'master'], check=True)
        st.write("Successfully committed and pushed the changes.")
    except subprocess.CalledProcessError as e:
        st.write(f"Error in Git operations: {e}")

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    
    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=texts, embedding=embeddings_model)
    db.persist()

    git_clone_and_commit_and_push(db)
