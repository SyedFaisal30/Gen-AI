from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://docs.chaicode.com/youtube/getting-started"
loader = WebBaseLoader(web_path=url)
soup = loader.scrape()

base_url = "https://chaidocs.vercel.app"
links = set()

for tag in soup.find_all("a", href=True):
    href = tag["href"]
    if href.startswith("/"):
        links.add(base_url + href)
    elif href.startswith("http") and href.startswith(base_url):
        links.add(href)
    
all_urls = [url] + list(links)
loader = WebBaseLoader(web_paths=all_urls)
docs = loader.load()
    
print("All Links of ChaiDocs are")
for u in all_urls:
    print(u)
print("Total Docs ", len(docs))
    
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100,
)
    
chunks = text_splitter.split_documents(docs)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_END_POINT = os.getenv("QDRANT_END_POINT")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API_KEY, 
    model="models/text-embedding-004" 
)

qclient = QdrantClient(
    url=QDRANT_END_POINT,
    api_key=QDRANT_API_KEY
)

vector_store = QdrantVectorStore.from_documents(
    url = QDRANT_END_POINT,
    api_key = QDRANT_API_KEY,
    collection_name = "chaiWeb",
    embedding = embedder,
    documents = chunks
)

print("Injection Done Successfully!!")