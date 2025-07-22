import os
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

config = {"configurable": {"thread_id":"1"}}

@tool()
def human_assistance_tool(query: str):
    """Request Assistance from a Human!"""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [human_assistance_tool]
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro", api_key = GEMINI_API_KEY)
llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: State):
    valid_messages = [msg for msg in state["messages"] if getattr(msg, "content", None)]
    if not valid_messages:
        raise ValueError("NO Valid Mesages with content found!.")
    print(valid_messages)
    
    message = llm_with_tools.invoke(valid_messages)
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph_with_checkpointer = create_chat_graph(checkpointer)

    while True:
        user_input = input("> ")
        for event in graph_with_checkpointer.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()
