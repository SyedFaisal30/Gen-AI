import os 
import json
import time
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langsmith.wrappers import wrap_openai
from langgraph.graph import StateGraph, START, END

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = wrap_openai(OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL
))

class State(TypedDict):
    user_message: str
    category: str
    ai_answer: str
    
class Classification(BaseModel):
    category: str
    
class Answer(BaseModel):
    answer: str
    
def classify_query(state: State):
    system_prompt = """
        You are an smart AI that Classisfies the user question into one of the following categories below:
        - coding
        - general_knowledge
        - sports
        - history
        - geography
        - science
        - technology
        - entertainment
        - politics
        
        Return only the category as lowercase string. 

        🧠 FORMAT TO FOLLOW STRICTLY:

        🚨 You MUST respond in this EXACT format, with no deviation. Never use markdown, bullet points, bold, or alternate styles.
            Only respond using:
            Step 1: ...
            Step 2: ...
            Step 3:
            Analyse: ...
            Think: ...
            Output: ...
            Validate: ...
            Result: ...
            
        
        Your job:
            - Detect the Category of the user's query 
            - Identify the Subcategory if possible
            - Respond with well-explained step-by-step answers using the structure above.
        Each part of Step 3 must contain a clear, complete sentence.

        Respond strictly in the above format every time. Do not include explanations outside the format.

    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_message"]}
    ]
    
    res = client.beta.chat.completions.parse(
        model= "gemini-2.0-flash",
        messages = messages,
        n = 1,
        response_format = Classification
    )
    
    state["category"] = res.choices[0].message.parsed.category
    return state

def route_by_category(state: State):
    return f"answer_{state['category']}_question"

def answer_coding_question(state: State):
    prompt = """
        You're a coding tutor AI. Follow strict reasoning:

            Step 1: Confirm category: coding  
            Step 2: Give it a Subcategory which coding question it is if language given then Javascript Question like this otherwise easy, Medium, Hard, etc
            Step 3:
            - Analyse: What is the problem about?
            - Think: What concepts are needed?
            - Output: Draft a code solution or explanation.
            - Validate: Does it logically work?
            - Result: Final explanation/code.

            Use all 5 sub-steps inside Step 3. Don't skip.
    """
    
    return ask_ai_with_prompt(state, prompt)

def answer_history_question(state: State):
    prompt = """
        You're a history expert AI.

            Step 1: Confirm category: history  
            Step 2: Give the query a subcategory on histry hat type of history itis.
            Step 3:
            - Analyse: What's the historical context?
            - Think: What facts/events matter?
            - Output: Construct the timeline or cause-effect chain.
            - Validate: Are the facts accurate?
            - Result: Clear historical explanation.

            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_geography_question(state: State):
    prompt = """
        You're a geography tutor AI.

            Step 1: Confirm category: geography  
            Step 2: Give the query a subcategory on what type of Geographical Question is.
            Step 3:
            - Analyse: What location/topic is being asked?
            - Think: Consider physical/cultural/economic features.
            - Output: Explain using spatial/geographic logic.
            - Validate: Check for real-world relevance.
            - Result: Coherent geographic explanation.
            
             Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_politics_question(state: State):
    prompt = """
        You're a political analyst AI.

            Step 1: Confirm category: politics  
            Step 2: Give the query a subcategory on Politics what type od Politics it is. 
            Step 3:
            - Analyse: What is the political issue?
            - Think: Consider systems, ideologies, current events.
            - Output: Lay out facts and perspectives.
            - Validate: Ensure neutrality and correctness.
            - Result: Balanced, informative answer.
            
            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_science_question(state: State):
    prompt = """
        You're a science explainer AI.

            Step 1: Confirm category: science 
            Step 2: Give the query a subcategory on Science that what type of Science question itis. 
            Step 3:
            - Analyse: Identify the scientific domain.
            - Think: Recall theories, formulas, logic.
            - Output: Use examples or analogies.
            - Validate: Check accuracy of principles.
            - Result: Scientifically sound explanation.
            
            Use all 5 sub-steps inside Step 3. Be accurate and structured.            
    """
    return ask_ai_with_prompt(state, prompt)

def answer_sports_question(state: State):
    prompt = """
        You're a sports expert AI.

            Step 1: Confirm category: sports  
            Step 2: Give the query a subcategory on what type of Sports question it is.
            Step 3:
            - Analyse: What sport and what aspect?
            - Think: Consider rules, players, rankings, events.
            - Output: Provide insights with stats if needed.
            - Validate: Check for factual accuracy.
            - Result: Engaging sports answer.
            
            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_technology_question(state: State):
    prompt = """
        You're a tech-savvy AI.

            Step 1: Confirm category: technology  
            Step 2: Give the query a subcategory on what type of Technology Question it is.
            Step 3:
            - Analyse: What's the tech topic?
            - Think: Consider innovations, impact, usage.
            - Output: Explain trends or technologies.
            - Validate: Ensure relevance and precision.
            - Result: Clear and updated tech insight.
            
            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_entertainment_question(state: State):
    prompt = """
        You're an entertainment critic AI.

            Step 1: Confirm category: entertainment  
            Step 2: Give the query a subcategory on Entertainment Question itis.
            Step 3:
            - Analyse: What show/movie/music is being discussed?
            - Think: Recall plot, genre, reception.
            - Output: Share opinions/facts/timelines.
            - Validate: Match known info or public perception.
            - Result: Fun, factual summary.

            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)

def answer_general_knowledge_question(state: State):
    prompt = """
        You're a general knowledge quizmaster.

            Step 1: Confirm category: general_knowledge  
            Step 2: Give the query a subcategory on what type of GK question itis.
            Step 3:
            - Analyse: What kind of GK fact is asked?
            - Think: Recall the correct domain.
            - Output: Fact-based explanation.
            - Validate: Check global/common knowledge validity.
            - Result: Final informative answer.

            Use all 5 sub-steps inside Step 3. Be accurate and structured.
    """
    return ask_ai_with_prompt(state, prompt)


# def ask_ai_with_prompt(state: State, system_prompt: str):
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": state["user_message"]}
#     ]
    
#     res = client.beta.chat.completions.parse(
#         model = "gemini-2.5-pro",
#         messages = messages,
#         # max_tokens = 2048,
#         response_format = Answer
#     )
#     state["ai_answer"] = res.choices[0].message.parsed.answer
#     return state

def ask_ai_with_prompt(state: State, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_message"]}
    ]

    res = client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=messages
    )

    state["ai_answer"] = res.choices[0].message.content
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("classify_query", classify_query)
graph_builder.add_node("route_by_category", route_by_category)

graph_builder.add_node("answer_coding_question", answer_coding_question)
graph_builder.add_node("answer_history_question", answer_history_question)
graph_builder.add_node("answer_geography_question", answer_geography_question)
graph_builder.add_node("answer_politics_question", answer_politics_question)
graph_builder.add_node("answer_science_question", answer_science_question)
graph_builder.add_node("answer_sports_question", answer_sports_question)
graph_builder.add_node("answer_technology_question", answer_technology_question)
graph_builder.add_node("answer_entertainment_question", answer_entertainment_question)
graph_builder.add_node("answer_general_knowledge_question", answer_general_knowledge_question)

graph_builder.add_edge(START, "classify_query"),
graph_builder.add_conditional_edges("classify_query", route_by_category)
graph_builder.add_edge("answer_coding_question", END)
graph_builder.add_edge("answer_history_question", END)
graph_builder.add_edge("answer_geography_question", END)
graph_builder.add_edge("answer_politics_question", END)
graph_builder.add_edge("answer_science_question", END)
graph_builder.add_edge("answer_sports_question", END)
graph_builder.add_edge("answer_technology_question", END)
graph_builder.add_edge("answer_entertainment_question", END)
graph_builder.add_edge("answer_general_knowledge_question", END)

graph = graph_builder.compile()

print("Ask me Anything😉! if want to Quit type Exit or Quit\n")

while True:
    user_imput = input("You: ")
    if user_imput.lower() in ["exit", "quit"]:
        print("HopeFully i helped you with you queries Good Bye until next time we will meet!")
        break
    
    state: State = {
        "user_message": user_imput,
        "category": "",
        "ai_answer": ""
    }
    
    result = graph.invoke(state)
    
    print(f"\nDetected Category: {result['category'].upper()}")
    print(f"🧠 Let's reason step-by-step:\n")

    try:
        parsed = json.loads(result["ai_answer"])
        
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(value, dict):
                    print(f"\n🧠 {key.upper()} \n")
                    for sub_key, sub_value in value.items():
                        print(f"   🔹 {sub_key.capitalize()}: {sub_value} \n")
        else:
            print("💬 \n", parsed)

    except json.JSONDecodeError:
        for line in result["ai_answer"].split("\n"):
            line = line.strip()
            if line:
                print("💬", line)
                time.sleep(0.3)
