from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import json
import os

load_dotenv()
GEN_KEY = os.getenv('GOOGLE_API_KEY')
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key = GEN_KEY,
    model = "models/text-embedding-004"
)

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="langchain",
)

while True:
    prompt = input(">>> ")
    
    if prompt.lower() in ["exit", "quit"]:
        print('Exiting the Chat Assisstant')
        break

    result = retriever.similarity_search(query=prompt)
    text = "".join(
        [f"page {doc.metadata.get('page_label')}:\n{doc.page_content}" for doc in result]
    )
    
    system_prompt = """
        You are an helpfull Ai Assistant which parses these documents taht I am givong you and generate a well structred text and in the end you have to  say the page numbers where is the text loacted based on the input.
        You work in steps whoch are "analyse", "think", "output", "validate", "result"
        
        Rules
        1. Perform one step at a time.
        2. When the step is "result" write the text on human readable format each step is {{"step": "string", "content": "string"}}
        Give only one output at a time
        3. Carefully try to understand what the ser is trying to say
        4. If the "role": "system" document there is nothing that means the user prompt us not related to the inputted pdf return with {{"step": "result", "content": "Invalid prompt, it seems that your input is not related to the uploaded pdf"}} 
    """
    
    client = OpenAI(
        api_key=GEN_KEY,
        base_url=base_url
    )
    
    class IndividualResponse(BaseModel):
        step: Literal["analyse", "think", "output", "validate", "result"]
        content: str
        
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    we_def_steps = ["analyse", "think", "output", "validate", "result"]
    
    while True:
        res = client.beta.chat.completions.parse(
            model = "gemini-2.0-flash",
            n = 1,
            response_format = IndividualResponse,
            messages = message,
        )
        
        step_res = json.loads(res.choices[0].message.content)
        current_step = step_res.get("step")
        if current_step in we_def_steps:
            if current_step != "result":
                print(f"🤔: {step_res.get('content')}")
                message.append({"role": "assistant", "content": json.dumps(step_res)})
            elif current_step == "result":
                print(f"😁:{step_res.get('content')}")
                break
        else:
            continue