from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from collections import defaultdict
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal
from openai import OpenAI
import json
import os

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_END_POINT = os.getenv("QDRANT_END_POINT")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API_KEY,
    model="models/text-embedding-004"
)

topic_keywords = {
    "html": ["html", "web", "tags", "markup"],
    "git": ["git", "branch", "commit", "repository", "stash", "github"],
    "devops": ["devops", "nginx", "vps", "docker", "rate limiting", "ssl", "postgresql", "logging"],
    "sql": ["sql", "database", "postgres", "normalization", "joins"],
    "django": ["django", "models", "templates", "forms", "jinja"],
    "c": ["c language", "c", "variables", "functions", "operators", "loops"]
}

def get_topic_from_query(query: str):
    query_lower = query.lower()
    for topic, keywords in topic_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return topic
    return None

def topic_routed_retrieve(query: str, top_k: int = 4, rrf_k: int = 60):
    topic = get_topic_from_query(query)

    retriever = QdrantVectorStore.from_existing_collection(
        url=QDRANT_END_POINT,
        api_key=QDRANT_API_KEY,
        collection_name="chaiWeb",
        embedding=embedder
    )

    print(f"\U0001F4D1 Topic Detected: {topic if topic else 'General'}")
    docs = retriever.similarity_search(query=query, k=rrf_k)

    doc_scores = defaultdict(float)
    seen_docs = {}
    for rank, doc in enumerate(docs[:top_k]):
        doc_id = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page_label', '')}"
        if doc_id not in seen_docs:
            seen_docs[doc_id] = doc
        doc_scores[doc_id] += 1.0 / (rank + rrf_k)

    sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
    ranked_result = [seen_docs[doc_id] for doc_id in sorted_doc_ids]

    return ranked_result, [doc.metadata.get("source", "") for doc in ranked_result]

class IndividualResponse(BaseModel):
    step: Literal["analyse", "think", "output", "validate", "result", "summary"]
    content: str

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=base_url
)

while True:
    prompt = input(">>> ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break

    retrieved_docs, sources = topic_routed_retrieve(prompt)

    if not retrieved_docs:
        print("\U0001F615 No relevant documents found.")
        continue

    full_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    system_prompt = """
        You are an expert AI assistant helping users understand technical concepts by synthesizing knowledge from documents.

        Perform these steps in order:
        1. analyse: 🧠 Identify and categorize the topic (e.g., HTML, Git, C, Django, SQL, DevOps)
        2. think: 🤔 Reflect on the query and relevant information
        3. output: 🎨 Provide a clear and useful explanation
        4. validate: ✅ Check accuracy and relevance
        5. summary: 🔄 Give a 2-3 line summary of the explanation
        6. result: 😊 Final answer plus related document links

        Rules:
        - Follow all 6 steps strictly.
        - Do not include unrelated information.
        - Present steps one by one.
        - Use concise and clear responses.
        - For unrelated queries, respond:
          {"step": "result", "content": "Invalid prompt, not related to the uploaded documents."}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {prompt}\n\nRelevant documents:\n{full_context}"}
    ]

    steps_order = ["analyse", "think", "output", "validate", "summary", "result"]

    while True:
        res = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            n=1,
            response_format=IndividualResponse,
            messages=messages,
        )

        step_res = json.loads(res.choices[0].message.content)
        current_step = step_res.get("step")

        if current_step in steps_order:
            if current_step != "result":
                emojis = {
                    "analyse": "🧠", "think": "🤔", "output": "🎨",
                    "validate": "✅", "summary": "🔄"
                }
                print(f"{emojis.get(current_step, '➡️')}: {step_res.get('content')}")
                messages.append({"role": "assistant", "content": json.dumps(step_res)})
            elif current_step == "result":
                links = "\n".join(f"🔗 {link}" for link in sources if link)
                print(f"\U0001F60A: {step_res.get('content')}\n\n📎 Relevant Links:\n{links if links else 'No links available.'}")
                break
        else:
            continue
