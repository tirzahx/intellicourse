from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
import json

from llm import llm
from vector_store import get_vector_store


class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str
    source_tool: str
    retrieved_context: str


@tool
def course_info_tool(query: str) -> dict:
    """Answers questions about university courses using the course catalog."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(k=4)
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = (
        f"You are Course ChatBot. Use the following context to answer the question:\n\n"
        f"Context: {context}\n\nQuestion: {query}"
    )
    response = llm.invoke(full_prompt)

    return {
        "answer": response.content,
        "context": context,
        "tool": "course_info_tool"
    }


@tool
def web_search_tool(query: str) -> dict:
    """Search the web for general knowledge questions."""
    tavily_tool = TavilySearch(name="web_search")
    raw_response = tavily_tool.run(query)

    if isinstance(raw_response, dict):
        if raw_response.get("answer"):
            answer = raw_response["answer"]
        elif raw_response.get("results"):
            answer = raw_response["results"][0]["content"]
        else:
            answer = str(raw_response)
    else:
        answer = str(raw_response)

    return {
        "answer": answer,
        "context": json.dumps(raw_response, indent=2) if isinstance(raw_response, dict) else str(raw_response),
        "tool": "web_search_tool"
    }


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
    """Executes the tool based on the classification and stores results."""
    question = state["messages"][-1].content
    classification = state["classification"]

    if classification == "course_info":
        result = course_info_tool(question)
    elif classification == "web_search":
        result = web_search_tool(question)
    else:
        result = {"answer": "Request Failed! Please rephrase.", "context": "", "tool": "none"}

    state["messages"].append(AIMessage(content=result["answer"]))
    state["source_tool"] = result["tool"]
    state["retrieved_context"] = result["context"]

    return state


# Build the graph
builder = StateGraph(State)

builder.add_node("router", route_question)
builder.add_node("execute_tool", execute_tool)

builder.add_edge(START, "router")
builder.add_edge("router", "execute_tool")
builder.add_edge("execute_tool", END)

graph = builder.compile()