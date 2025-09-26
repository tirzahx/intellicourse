from typing import TypedDict, Annotated, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
import json

from llm import llm
from vector_store import get_vector_store

class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str

@tool
def course_info_tool(query: str) -> str:
    """Answers questions about university courses using the course catalog.
    Use this for questions about course schedules, prerequisites, and descriptions.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(k=4)
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = (f"You are Course ChatBot. Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {query}")
    response = llm.invoke(full_prompt)
    return response.content

@tool
def web_search_tool(query: str) -> str:
    """Search the web for general knowledge questions."""
    tavily_tool = TavilySearch(name="web_search")
    raw_response = tavily_tool.run(query)

    if isinstance(raw_response, dict):
        if raw_response.get("answer"):
            return raw_response["answer"]
        elif raw_response.get("results"):
            return raw_response["results"][0]["content"]
    
    return str(raw_response)

def route_question(state: State) -> State:
    """Uses the LLM to classify the user's question."""
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at classifying user questions.
    Classify the user's final question as either 'course_info' or 'web_search'.
    'course_info' is for questions about university courses (e.g., prerequisites, schedules, descriptions).
    'web_search' is for general knowledge questions (e.g., "who is Ada Lovelace?").
    
    Your response must be ONLY 'course_info' or 'web_search'.
    
    Question: {question}
    
    Classification:""")
    
    last_message = state["messages"][-1].content
    classification_chain = prompt | llm
    
    classification_result = classification_chain.invoke({"question": last_message}).content
    
    state["classification"] = classification_result.strip().lower()
    return state

def execute_tool(state: State) -> State:
    """Executes the tool based on the classification."""
    question = state["messages"][-1].content
    classification = state["classification"]
    
    if classification == "course_info":
        result = course_info_tool(question)
    elif classification == "web_search":
        result = web_search_tool(question)
    else:
        result = "Request Failed! Please rephrase."
        
    state["messages"].append(AIMessage(content=result))
    return state

def summarize_answer(state: State) -> State:
    """Uses the LLM to summarize the retrieved content into a concise answer."""
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Provide a very short and direct one-to-two line answer to the user's question.
    Use ONLY the following context to generate your response. If the context does not contain the answer, say so.
    
    Question: {question}
    
    Context: {context}
    
    Answer:""")
    
    question = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
    context = [msg.content for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
    
    summary_chain = prompt | llm
    final_answer = summary_chain.invoke({"question": question, "context": context}).content
    
    state["messages"].append(AIMessage(content=final_answer))
    return state
    
builder = StateGraph(State)

builder.add_node("router", route_question)
builder.add_node("execute_tool", execute_tool)
builder.add_node("summarize_answer", summarize_answer)

builder.add_edge(START, "router")
builder.add_edge("router", "execute_tool")

builder.add_conditional_edges(
    "execute_tool",
    lambda state: "summarize_web_search" if state.get("classification") == "web_search" else "__end__",
    {
        "summarize_web_search": "summarize_answer",
        "__end__": END
    }
)

builder.add_edge("summarize_answer", END)

graph = builder.compile()