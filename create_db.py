from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

os.environ['OPENAI_API_KEY'] = '[your-api-key]'

loader = PyPDFLoader("[your_manual.pdf]")
pages = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings()
persist_directory = './[folder_name]'
db = Chroma.from_documents(documents=texts, embedding=embeddings_model, persist_directory=persist_directory)
db.persist()
