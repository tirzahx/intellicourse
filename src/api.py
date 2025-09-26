import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from graph import graph
from langchain_core.messages import AIMessage

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_tool: Optional[str] = None
    retrieved_context: Optional[str] = None

app = FastAPI(
    title="IntelliCourse API")

@app.post("/chat", response_model=QueryResponse)
def chat_with_agent(request: QueryRequest):
    
    final_state = graph.invoke({"messages": [("user", request.query)]})
    answer = final_state["messages"][-1].content
    return QueryResponse(answer=answer)

#to run- uvicorn api:app --reload