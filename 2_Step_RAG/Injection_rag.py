from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GEN_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GEN_KEY)

path = Path(__file__).parent/"nodejs.pdf"

loader = PyPDFLoader(file_path = path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

splitted_docs = splitter.split_documents(documents = docs)

embedder = GoogleGenerativeAIEmbeddings(google_api_key=GEN_KEY, model="models/text-embedding-004")

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="langchain",
    embedding=embedder
)

vector_store.add_documents(documents=splitted_docs)
print("Injection Phase Done")