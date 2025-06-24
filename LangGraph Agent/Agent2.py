import os
from typing import TypedDict, List, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq


api = os.environ.get("GROQ_API_KEY")

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

def process(state: AgentState) -> AgentState:
    """ This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

converstional_his = []

user_input = input("Enter: ")
while user_input != "Exit":
    converstional_his.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": converstional_his})  
    converstional_his = result["messages"]
    user_input = input("Enter: ")
    