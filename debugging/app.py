import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "ReActAgent"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY)


def make_default_graph():
    def call_model(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    agent = builder.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    model_with_tools = llm.bind_tools([add])

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("llm", call_model)
    graph_workflow.add_node("tools", ToolNode(tools=[add]))
    graph_workflow.add_edge(START, "llm")
    graph_workflow.add_conditional_edges("llm", tools_condition)
    graph_workflow.add_edge("tools", "llm")

    agent = graph_workflow.compile()
    return agent


agent = make_alternative_graph()
