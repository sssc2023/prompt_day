__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
import subprocess


#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def git_clone_and_commit_and_push():
    try:
        subprocess.run(['git', 'clone', 'https://github.com/sssc2023/prompt_day.git'], check=True)
        subprocess.run(['git', 'add', 'db'], cwd='prompt_day', check=True)  # 'db'가 실제 추가하려는 파일 또는 폴더 이름이라고 가정
        subprocess.run(['git', 'commit', '-m', 'Add generated file'], cwd='prompt_day', check=True)
        subprocess.run(['git', 'push', 'origin', 'master'], cwd='prompt_day', check=True)
    except subprocess.CalledProcessError as e:
        st.write(f"Git 명령어 실행 중 에러 발생: {e}")

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)
    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(documents=texts, embedding=embeddings_model)
    db.persist()

    git_clone_and_commit_and_push()
