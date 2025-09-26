import os
from dotenv import load_dotenv
from graph import graph
from vector_store import get_vector_store

load_dotenv()

def main():
    print("IntelliCourse is ready! Ask me about courses or any general question. Type 'exit' to quit.")
    
    while True:
        query = input("Enter question: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        final_state = graph.invoke({"messages": [("user", query)]})
        answer = final_state["messages"][-1].content
        
        print(f"\Answer: {answer}\n")

if __name__ == "__main__":
    main()