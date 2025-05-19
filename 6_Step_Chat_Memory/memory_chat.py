from mem0 import Memory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_END_POINT = os.getenv("QDRANT_END_POINT")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = "bolt://localhost:7687"
GEMINI_PROXY = "http://localhost:3000/v1"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": GEMINI_API_KEY,
            "base_url": GEMINI_BASE_URL,
            "model": "models/embedding-004"
        },
    },
    "llm":{
        "provider": "openai",
        "config": {
            "api_key": GEMINI_API_KEY,
            "base_url": GEMINI_BASE_URL,
            "model": "gemini-2.0-flash"
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "api_key": QDRANT_API_KEY,
            "url": QDRANT_END_POINT,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USER,
            "password": NEO4J_PASS
        },
    },
}

mem_client = Memory.from_config(config)

def chat(message):
    mem_result = mem_client.search(query=message, user_id="faisal30")
    
    memories = "\n".join([m["memory"] for m in mem_result.get("results", [])])
    
    print(f"\n\nMEMORY: \n\n{memories}\n\n")
    
    system_prmpt = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to systematically analyze input content, extract structured knowledge, and maintain an optimized memory store. Your primary function is information distillation and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling

        Memory and Score:
        {memories}
    """
    
    messages = [
        {"role": "system", "content": system_prmpt},
        {"role": "user", "content": message}
    ]
    
    result = mem_client.llm.chat(messages)
    
    assistant_reply = result ["content"]
    messages.append({"role": "assistant", "content": assistant_reply})
    
    mem_client.add(messages, user_id="faisal30")
    
    return assistant_reply

if __name__ == "__main__":
    while True:
        prompt = input(">> ")
        if prompt.lower() in ["exit", "quit"]:
            break
        print("🤖: ", chat(prompt))