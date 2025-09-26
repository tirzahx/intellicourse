import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "courses"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension = 384

if not pc.has_index(index_name):
    print(f"Index '{index_name}' does not exist. Creating...")
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def get_vector_store():
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)

def load_and_ingest_pdfs():
    
    pdf_files = [
        "data/BA_Catalog.pdf",
        "data/CS_Catalog.pdf",
        "data/Eng_Catalog.pdf",
        "data/Lit_Catalog.pdf",
        "data/Psy_Catalog.pdf"
    ]

    all_documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents = loader.load()
        for d in documents:
            d.metadata["source"] = file
        all_documents.extend(documents)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_documents)
    
    PineconeVectorStore.from_documents(
        index_name=index_name,
        embedding=embeddings,
        documents=chunks
    )
    print("Successfully added all document chunks into Pinecone.")

if __name__ == "__main__":
    try:
        load_and_ingest_pdfs()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")