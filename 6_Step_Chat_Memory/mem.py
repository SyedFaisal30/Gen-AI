import os
import json
import uuid
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
from neo4j import GraphDatabase

# === 1. Load Env ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_END_POINT")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = "bolt://localhost:7687"

# === 2. Init Embedding Model ===
embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=GOOGLE_API_KEY,
    model="models/embedding-001"
)

# === 3. Init Qdrant ===
qdrant_client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "memory"
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
vector_store = Qdrant(client=qdrant_client, collection_name=COLLECTION_NAME, embeddings=embedding_model)

# === 4. Init Gemini via OpenAI-Compatible Interface ===
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# === 5. Init Neo4j ===
neo4j_driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))

def insert_into_neo4j(triples):
    with neo4j_driver.session() as session:
        for subj, pred, obj in triples:
            session.run("""
                MERGE (a:Entity {name: $subj})
                MERGE (b:Entity {name: $obj})
                MERGE (a)-[:RELATION {type: $pred}]->(b)
            """, subj=subj, pred=pred, obj=obj)

def extract_triples_from_text(text):
    prompt = f"""
    Extract factual knowledge triples from the text below.
    Format: (Subject, Predicate, Object)
    Only output one triple per line. No commentary.

    \"\"\"{text}\"\"\"
    """
    response = client.chat.completions.create(
        model="gemini-2.0",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    triples = []
    for line in lines:
        line = line.strip("() ")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples

# === 6. Store in Memory (Qdrant) and Neo4j ===
def store_message(user_message, response_message):
    doc_id = str(uuid.uuid4())
    doc = Document(page_content=user_message, metadata={"id": doc_id})
    vector_store.add_documents([doc])
    triples = extract_triples_from_text(user_message)
    if triples:
        insert_into_neo4j(triples)

# === 7. Retrieve Memories ===
def retrieve_from_qdrant(query, top_k=5):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

# === 8. Structured System Prompt (Step-by-Step) ===
def build_structured_prompt(query, context):
    return [
        {
            "role": "system",
            "content": """
You are a memory-aware AI assistant that reasons step-by-step.

Follow these stages:
1. analyse: 🧠 Identify and categorize the topic
2. think: 🤔 Reflect on the query + memory
3. output: 🎨 Provide a detailed explanation
4. validate: ✅ Check accuracy & logic
5. summary: 🔄 Give a 2-3 line recap
6. result: 😊 Final polished answer

Do not skip steps. Use the retrieved memory as context but verify before using.
"""
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nMemory:\n{context}"
        }
    ]

# === 9. Chat Loop ===
def chat(query):
    context = retrieve_from_qdrant(query)
    messages = build_structured_prompt(query, context)

    steps = ["analyse", "think", "output", "validate", "summary", "result"]
    seen_steps = set()

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages,
            n=1
        )

        reply = json.loads(response.choices[0].message.content)
        step = reply.get("step")
        content = reply.get("content")

        if step in seen_steps:
            continue  # avoid duplicate step
        seen_steps.add(step)

        print(f"{step.title()}: {content}")
        messages.append({"role": "assistant", "content": json.dumps(reply)})

        if step == "result":
            store_message(query, content)
            break

# === 10. CLI ===
if __name__ == "__main__":
    print("🤖 Gemini Chat (Qdrant + Neo4j + Step Reasoning)")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        chat(user_input)
