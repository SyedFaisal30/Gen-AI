from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import json
import os

load_dotenv()

GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_KEY,
    model="models/text-embedding-004"
)

collection_names = ["next", "python", "node", "react"]

retrievers = {
    name: QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url="https://c2e95f80-97aa-42fa-844e-df5bf2d2677a.europe-west3-0.gcp.cloud.qdrant.io",
        api_key=QDRANT_API_KEY,
        collection_name=name,
    )
    for name in collection_names
}

def semantic_based_router(query: str, top_k: int = 3, top_collections: int = 4):
    """
    Perform semantic routing by checking similarity scores across collections.
    Returns top N collections most relevant to the query.
    """
    collection_scores = {}

    for collection, retriever in retrievers.items():
        docs = retriever.similarity_search(query=query, k=top_k)
        if not docs:
            collection_scores[collection] = 0
            continue
        
        score = 0
        for rank, doc in enumerate(docs, start=1):
            score += 1 / rank
        collection_scores[collection] = score

    sorted_collections = sorted(collection_scores.items(), key=lambda x: x[1], reverse=True)
    top_collections_list = [col for col, score in sorted_collections if score > 0][:top_collections]

    if not top_collections_list:
        print("⚠️ No collections found by semantic routing, searching all.")
        return list(collection_names)

    print(f"🧠 Semantic routing matched collections: {top_collections_list}")
    return top_collections_list

def fanout_retrieve_semantic(query: str, top_k: int = 4):
    """Retrieve docs from top semantically matched collections."""
    matched_collections = semantic_based_router(query, top_k=3, top_collections=4)
    seen_docs = {}

    for name in matched_collections:
        retriever = retrievers[name]
        docs = retriever.similarity_search(query=query, k=top_k)
        for doc in docs:
            doc_id = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page_label', '')}:{name}"
            if doc_id not in seen_docs:
                seen_docs[doc_id] = doc
                doc.metadata["collection"] = name

    ranked_result = list(seen_docs.values())

    for col in matched_collections:
        pages = [doc.metadata.get('page_label', 'N/A') for doc in ranked_result if doc.metadata.get('collection') == col]
        print(f"📄 Collection '{col}' has data in pages: {sorted(set(pages))}")

    return ranked_result

class IndividualResponse(BaseModel):
    step: Literal["analyse", "think", "output", "validate", "result"]
    content: str

client = OpenAI(
    api_key=GEMINI_KEY,
    base_url=base_url
)

print("Semantic routing assistant started. Type your query or 'exit' to quit.")

while True:
    prompt = input(">>> ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break

    retrieved_docs = fanout_retrieve_semantic(prompt)

    if not retrieved_docs:
        print("😕 No relevant documents found.")
        continue

    full_context = "\n\n".join([
        f"From Collection '{doc.metadata.get('collection', 'unknown')}', "
        f"page {doc.metadata.get('page_label', 'N/A')} ({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    system_prompt = """
        You are an expert AI assistant helping users understand technical concepts by synthesizing knowledge from multiple documents.
        You must strictly follow the 5 steps in order, outputting exactly one step at a time.
        Work in the following steps:
        1. analyse: Review the query and categorize it (e.g., Python, React, node, nextjs collections etc.)
        2. think: Reflect on which collection has the best information
        3. output: Draft a useful explanation based on the top documents
        4. validate: Ensure the explanation is accurate and specific
        5. result: Final formatted output with sources and page numbers

        Rules:
        - Be concise and clear.
        - You must complete all 5 steps. Do not skip any step.
        - Do not include content from unrelated documents.
        - Cite exact pages and document sources at the end.
        - If the Question is not related to these four topics then say:
        {"step": "result", "content": "Invalid prompt, not related to the uploaded documents."}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {prompt}\n\nRelevant documents:\n{full_context}"}
    ]

    steps_order = ["analyse", "think", "output", "validate", "result"]

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
                print(f"🤔: {step_res.get('content')}")
                messages.append({"role": "assistant", "content": json.dumps(step_res)})
            elif current_step == "result":
                print(f"😁: {step_res.get('content')}")
                break
        else:
            continue
