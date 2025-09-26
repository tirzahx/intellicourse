import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

load_dotenv()

llm = ChatGroq(  
    model="llama-3.1-8b-instant",  
    temperature=0,
)

base_prompt = """
You are Course ChatBot, an assistant for answering student questions about courses.
Use the provided course catalog context to answer.
If the answer is not found, say "I don't know".
"""