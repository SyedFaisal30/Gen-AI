import os
import time
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langsmith.wrappers import wrap_openai
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Gemini-compatible OpenAI wrapper
client = wrap_openai(OpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL))

# State schema
class State(TypedDict):
    user_message: str
    category: str
    ai_answer: str

# Response schemas
class Classification(BaseModel):
    category: str

class Answer(BaseModel):
    answer: str

# Step 1: Classify question
def classify_query(state: State):
    user_message = state["user_message"]
    system_prompt = """
You are a smart AI that classifies user questions into only ONE of the following categories:
- coding
- general_knowledge
- sports
- history
- geography
- science
- technology
- entertainment
- politics

Return exactly one category in lowercase.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        messages=messages,
        response_format=Classification
    )
    state["category"] = res.choices[0].message.parsed.category
    return state

# Step 2: Route based on category
def route_by_category(state: State):
    return "answer_coding_question" if state["category"] == "coding" else "answer_general_question"

# Strict CoT Prompt
def strict_cot_prompt(user_message: str, category: str) -> list:
    return [
        {"role": "system", "content": f"""
You are a strict Chain-of-Thought reasoning assistant.

Follow these **exact labeled steps** — no combining or skipping. Use each step as shown:

- Step 1: Confirm the category as "{category}"
- Step 2: Deep reasoning split into 5 parts:
    - Analyse: Briefly analyze what the user is really asking.
    - Think: Think through what concepts, data, or logic is needed.
    - Output: Draft a rough outline of the answer.
    - Validate: Validate the logic and facts used in your output.
    - Result: Final, coherent and clear answer summary.

- Step 3: Re-state the user intent.
- Step 4: Elaborate on related global or contextual facts (optional if covered).
- Step 5: Final statement conclusion (if different from Result).

⚠️ IMPORTANT: Use all 5 sub-steps under Step 2 and label them exactly. No merging steps. Be verbose, reflective, and structured.
"""},
        {"role": "user", "content": user_message}
    ]

# Step 3A: Answer coding questions
def answer_coding_question(state: State):
    messages = strict_cot_prompt(state["user_message"], "coding")
    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        messages=messages,
        max_tokens=2048,
        response_format=Answer
    )
    state["ai_answer"] = res.choices[0].message.parsed.answer
    return state

# Step 3B: Answer general questions
def answer_general_question(state: State):
    messages = strict_cot_prompt(state["user_message"], state["category"])
    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        messages=messages,
        max_tokens=2048,
        response_format=Answer
    )
    state["ai_answer"] = res.choices[0].message.parsed.answer
    return state

# Graph setup
graph_builder = StateGraph(State)
graph_builder.add_node("classify_query", classify_query)
graph_builder.add_node("route_by_category", route_by_category)
graph_builder.add_node("answer_coding_question", answer_coding_question)
graph_builder.add_node("answer_general_question", answer_general_question)
graph_builder.add_edge(START, "classify_query")
graph_builder.add_conditional_edges("classify_query", route_by_category)
graph_builder.add_edge("answer_coding_question", END)
graph_builder.add_edge("answer_general_question", END)
graph = graph_builder.compile()

# ✨ Chat Loop
print("🤖 Ask me anything! (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye! See you next time.")
        break

    state: State = {
        "user_message": user_input,
        "category": "",
        "ai_answer": ""
    }

    result = graph.invoke(state)

    print(f"\n🤖 AI is thinking...\n")
    time.sleep(1.2)

    print(f"📂 Detected Category: {result['category'].upper()}")
    print(f"🔎 Let's go through it step-by-step:\n")

    for line in result["ai_answer"].split("\n"):
        if line.strip().lower().startswith("step"):
            print(f"🧠 {line}")
        elif line.strip():
            print(f"💬 {line}")
        time.sleep(0.7)

    print("\n" + "-" * 60 + "\n")
