from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import json
import os

load_dotenv()

GEMINI_KEY = os.getenv('GOOGLE_API_KEY')
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key = GEMINI_KEY,
    model = "models/text-embedding-004"
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

def fanout_retrieve(query, top_k: int = 4):
    results = []
    for name, retriever in retrievers.items():
        docs = retriever.similarity_search(query=query, k=top_k)
        for doc in docs:
            doc.metadata["collection"] = name
            results.append(doc)
    return results
class IndividualResponse(BaseModel):
        step: Literal["analyse", "think", "output", "validate", "result"]
        content: str
        
while True:
    prompt = input(">>> ")
    
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the Chat Assistant")
        break
    
    retrieved_docs = fanout_retrieve(prompt)
    
    if not retrieved_docs:
        print("😕 No relevant documents found.")
        break
    
    # result = fanout_retrieve(prompt, top_k=4)
    # text = "".join(
    #     [f"page {doc.metadata.get('page_label', 'N/A')} ({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}" for doc in result]
    # )
    full_context = "\n\n".join([
        f"From collection '{doc.metadata.get('collection', 'unknown')}', "
        f"page {doc.metadata.get('page_label', 'N/A')} ({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    
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
        - If the Question is not trelated to these four topics then say sorryi can't assist you on this topic.
        If the question is not related to the collections, say:
        {"step": "result", "content": "Invalid prompt, not related to the uploaded documents."}
    """
    
    client = OpenAI(
        api_key=GEMINI_KEY,
        base_url=base_url
    )
        
    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": full_context }
    ]
    
    steps_order = ["analyse", "think", "output", "validate", "result"]
    
    while True:
        res = client.beta.chat.completions.parse(
            model = "gemini-2.0-flash",
            n = 1,
            response_format = IndividualResponse,
            messages = messages,
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