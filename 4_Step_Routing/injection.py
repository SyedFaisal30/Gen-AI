from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_KEY)

path = Path(__file__).parent.parent/"python.pdf"

loader = PyPDFLoader(file_path = path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

splitted_docs = splitter.split_documents(documents = docs)

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_KEY,
    model="models/text-embedding-004"
)

vector_store = QdrantVectorStore.from_documents(
    documents = [],
    url = os.getenv("QDRANT_END_POINT"),
    api_key = os.getenv("QDRANT_API_KEY"),
    collection_name = "python",
    embedding=embedder
)

vector_store.add_documents(documents=splitted_docs)
print("Collection Created Successfully!")