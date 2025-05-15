from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import json
import os
from collections import defaultdict

load_dotenv()

GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_KEY,
    model = "models/text-embedding-004"
)

collection_names = ["node", "react", "next", "python"]

retrievers = {
    name: QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url = "https://c2e95f80-97aa-42fa-844e-df5bf2d2677a.europe-west3-0.gcp.cloud.qdrant.io",
        api_key = os.getenv("QDRANT_API_KEY"),
        collection_name = name,
    )
    for name in collection_names
}

collection_keywords = {
    "node": ["nodejs", "node", "express", "javascript", "backend", "server"],
    "next": ["nextjs", "react", "next", "framework", "javascript", "ssr", "ssg"],
    "python": ["python", "backend", "script", "django", "flask"],
    "react": ["react", "jsx", "hooks", "component", "frontend"]
}

def logic_based_router(query: str):
    """Return list of relevant colections by checking keywords"""
    query_lower = query.lower()
    matched_collections = set()
    
    for collection, keywords in collection_keywords.items():
        for kw in keywords:
            if kw in query_lower:
                matched_collections.add(collection)
                break
    
    if not matched_collections:
        print("⚠️ No specific collection matched by logic, searching all.")
        return list(collection_names)
    
    print(f"🗂️ Logic-based routing matched collections: {sorted(matched_collections)}")
    return sorted(matched_collections)

def fanout_retrieve_with_rrf(query: str, top_k: int = 4, rrf_k: int = 60):
    """Perform retrieval on logic-filtered collections with reciprocal rank fusion."""
    matched_collections = logic_based_router(query)
    ranked_results = []
    doc_scores = defaultdict(float)
    seen_docs = {}
    
    for name in matched_collections:
        retriever = retrievers[name]
        docs = retriever.similarity_search(query=query, k=top_k)
        for rank, doc in enumerate(docs):
            doc_id = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page_label', '')}:{name}"
            if doc_id not in seen_docs:
                seen_docs[doc_id] = doc
                doc.metadata["collection"] = name
            doc_scores[doc_id] += 1.0 / (rank + rrf_k)
            
    sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
    ranked_result = [seen_docs[doc_id] for doc_id in sorted_doc_ids]

    # Print which pages were found in each collection
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

while True:
    prompt = input(">>> ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break

    retrieved_docs = fanout_retrieve_with_rrf(prompt)

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
        1. analyse: Review the query and categorize it (e.g., Python, React, langchain, nextjs collections etc.)
        2. think: Reflect on which collection has the best information
        3. output: Draft a useful explanation based on the top documents
        4. validate: Ensure the explanation is accurate and specific
        5. result: Final formatted output with sources and page numbers

        Rules:
        - You must complete all 5 steps. Do not skip any step.
        - Be concise and clear.
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