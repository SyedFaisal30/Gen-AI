from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from collections import defaultdict
import os
import json

load_dotenv()

GEMINI_KEY = os.getenv('GOOGLE_API_KEY')
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Initialize Embeddings and VectorStores
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_KEY,
    model="models/text-embedding-004"
)

collection_names = ["langchain", "nextjs", "python", "react"]

retrievers = {
    name: QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url="http://localhost:6333",
        collection_name=name
    )
    for name in collection_names
}

client = OpenAI(
    api_key=GEMINI_KEY,
    base_url=base_url
)

class IndividualResponse(BaseModel):
    step: Literal["analyse", "think", "output", "validate", "result"]
    content: str
    
def generate_hypothetical_answer(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert assistant. Given a user question, generate a hypothetical answer or summary as if you were answering it yourself without any documents",
        },
        {
            "role": "user",
            "content": f"Question: {query}"
        }
    ]
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )
    return response.choices[0].message.content

def hyde_fanout_retrieve(query: str, top_k: int = 4):
    hypo_answer = generate_hypothetical_answer(query)
    query_embedding = embedder.embed_query(hypo_answer)

    all_docs = []
    for name, retriever in retrievers.items():
        docs = retriever.similarity_search_by_vector(query_embedding, k=top_k)
        for doc in docs:
            doc.metadata["collection"] = name
        all_docs.extend(docs)

    return all_docs

while True:
    prompt = input(">>> ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break

    retrieved_docs = hyde_fanout_retrieve(prompt)

    if not retrieved_docs:
        print("😕 No relevant documents found.")
        continue

    # Compose full context string
    full_context = "\n\n".join([
        f"From Collection '{doc.metadata.get('collection', 'unknown')}', "
        f"page {doc.metadata.get('page_label', 'N/A')} "
        f"({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Reasoning Prompt
    system_prompt = """
        You are an expert AI assistant helping users understand technical concepts by synthesizing knowledge from multiple documents.
        Work in the following steps:
        1. analyse: Review the query and categorize it (e.g., Python, React, langchain, nextjs collections etc.)
        2. think: Reflect on which collection has the best information
        3. output: Draft a useful explanation based on the top documents
        4. validate: Ensure the explanation is accurate and specific
        5. result: Final formatted output with sources and page numbers

        Rules:
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
            else:
                print(f"😁: {step_res.get('content')}")
                break
        else:
            continue