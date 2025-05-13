from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os
import json

# Load environment variables
load_dotenv()

GEMINI_KEY = os.getenv('GOOGLE_API_KEY')
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Initialize embedding model
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_KEY,
    model="models/text-embedding-004"
)

# Vector store collections
collection_names = ["langchain", "nextjs", "python", "react"]

retrievers = {
    name: QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url="http://localhost:6333",
        collection_name=name
    )
    for name in collection_names
}

# Initialize Gemini client via OpenAI-compatible API
client = OpenAI(
    api_key=GEMINI_KEY,
    base_url=base_url
)

# Response format for each step
class IndividualResponse(BaseModel):
    step: Literal["analyse", "think", "output", "validate", "result"]
    content: str

# Step-back abstraction: multiple broader versions
def step_back_prompt(query: str) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": "You are a senior research assistant. Given a user query, generate 2 to 3 broader or more general abstracted forms of the query. These should capture the underlying intent or topic, not just paraphrase. Return them as a bullet list."
        },
        {
            "role": "user",
            "content": f"User Query: '{query}'\n\nGenerate 2-3 broader versions."
        }
    ]
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )
    bullet_text = response.choices[0].message.content.strip()
    return [line.lstrip("-• ").strip() for line in bullet_text.splitlines() if line.strip()]

# Retrieve relevant documents using all step-back queries
def stepback_retrieve(query: str, top_k: int = 4):
    abstracted_queries = step_back_prompt(query)
    print(f"\n🧠 Step-Back Abstracted Queries:")
    for aq in abstracted_queries:
        print(f"- {aq}")

    all_docs = []
    for abs_query in abstracted_queries:
        query_embedding = embedder.embed_query(abs_query)
        for name, retriever in retrievers.items():
            docs = retriever.similarity_search_by_vector(query_embedding, k=top_k)
            for doc in docs:
                doc.metadata["collection"] = name
                doc.metadata["abstracted_query"] = abs_query
            all_docs.extend(docs)

    return abstracted_queries, all_docs

# Main interaction loop
while True:
    prompt = input(">>> ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break

    # Get abstracted queries and retrieved documents
    stepback_queries, retrieved_docs = stepback_retrieve(prompt)
    if not retrieved_docs:
        print("😕 No relevant documents found.")
        continue

    # Combine document content
    full_context = "\n\n".join([
        f"From Collection '{doc.metadata.get('collection', 'unknown')}', "
        f"page {doc.metadata.get('page_label', 'N/A')} "
        f"({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Format step-back query text
    stepback_query_text = "\n".join([f"- {q}" for q in stepback_queries])

    # System prompt
    system_prompt = f"""
You are an expert AI assistant helping users understand technical concepts using retrieved documents.
You must strictly follow the 5 steps in order, outputting exactly one step at a time.

Steps:
1. analyse: Categorize the original and abstracted queries.
2. think: Decide which document collections and queries best address the user’s intent.
3. output: Use the selected documents to generate an informative and helpful answer.
4. validate: Check for technical accuracy and whether the response aligns with user needs.
5. result: Return the final formatted answer with document citations (include source and page).

Rules:
- You must complete all 5 steps. Do not skip any step.
- Output must be in JSON with this format: {{ "step": "stepname", "content": "..." }}
- In step "result", cite the sources using document metadata.
- If the query is irrelevant to the documents, return: {{ "step": "result", "content": "Invalid prompt, not related to the uploaded documents." }}
"""

    # Chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Original Query: {prompt}\n\nStep-Back (Abstracted):\n{stepback_query_text}\n\nRelevant documents:\n{full_context}"
        }
    ]

    # Step-by-step reasoning loop
    steps_order = ["analyse", "think", "output", "validate", "result"]

    while True:
        res = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            response_format=IndividualResponse,
            messages=messages,
        )

        step_res = json.loads(res.choices[0].message.content)
        current_step = step_res.get("step")

        if current_step in steps_order:
            if current_step != "result":
                print(f"🤔 [{current_step.upper()}]: {step_res.get('content')}")
                messages.append({"role": "assistant", "content": json.dumps(step_res)})
            else:
                print(f"\n✅ Final Answer:\n{step_res.get('content')}\n")
                break
        else:
            continue
