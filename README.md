## **IntelliCourse**

IntelliCourse is an intelligent agent designed to answer a wide range of questions using a combination of a university course catalog and a web search tool. Built with **LangGraph**, the agent intelligently routes queries to the most relevant information source—a local knowledge base (RAG) for course-specific questions and a web search for general knowledge. The agent is exposed as a REST API using **FastAPI** for seamless integration into other applications.

### **Features**

* **Intelligent Routing**: Classifies user queries and routes them to the correct tool.
* **Course Catalog Q&A**: Answers specific questions about university courses using a private, embedded knowledge base.
* **Web Search Capability**: Handles general knowledge questions by performing a real-time web search.
* **REST API**: Provides a robust, standard API for external use.

### **Project Setup**

#### **1. Clone the Repository**

First, clone this repository to your local machine:

```bash
git clone [https://github.com/your-username/IntelliCourse.git](https://github.com/your-username/IntelliCourse.git)
cd IntelliCourse
```

#### **2. Set up a Python Virtual Environment (Recommended)**

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### **3. Install Dependencies**

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### **4. Configure API Keys**

This project requires API keys for services: Pinecone, Tavily and Groq (or your chosen LLM).
Create a file named `.env` in the root of your project and add the following keys:

```bash
PINECONE_API_KEY="your-pinecone-api-key"
TAVILY_API_KEY="your-tavily-api-key"
GROQ_API_KEY="your-groq-api-key"
```

#### **5. Ingest Data**

Before using the agent, you must load the course catalog PDFs into your Pinecone vector store.
Place your PDF files inside the `data/` directory. Then, from the root of your project, run the ingestion script as a module:

```bash
python -m src.vector_store
```

You should see a message confirming the documents were successfully added to your Pinecone index.

### **How to Run the API**

To start the FastAPI server, run the following command from the project root:

```bash
python -m uvicorn src.api:app --reload
```
The server will be accessible at `http://127.0.0.1:8000`.

#### **`POST /chat`**

The `/chat` endpoint is the main interface for interacting with the agent. It accepts a user query and returns a generated answer.

* **URL**: `http://127.0.0.1:8000/chat`
* **Method**: `POST`
* **Content-Type**: `application/json`

### **Sample Query and Response**

You can test the endpoint using Swagger UI or a Python script.

#### **Query**

**Using Python's `requests` library**

```bash
import requests
import json

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "application/json"}
data = {"query": "Prerequisites for ENGR 102?"}
response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
```
#### **Response**

A successful response will return a JSON object containing the agent's answer.

```bash
{
    'answer': 'The prerequisites for ENGR 102 are High school calculus.', 'source_tool': None, 'retrieved_context': None
}
```

#### **Screenshot (using Swagger UI)**

![Swagger UI Screenshot](screenshots\post_query.png)