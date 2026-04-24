import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

load_dotenv()

class MemoryAgent:
    def __init__(self, user_id: str = "default_user", max_tokens: int = 4000):
        self.user_id = user_id
        self.max_tokens = max_tokens
        
        # Build and compile the LangGraph workflow
        self.graph = build_graph()
        
        # Initialize internal state to keep chat history (acting as short term memory)
        self.state = {
            "messages": [],
            "user_id": self.user_id,
            "memory_budget": self.max_tokens
        }

    def chat(self, user_query: str) -> str:
        # Add the new human message to the state
        new_msg = HumanMessage(content=user_query)
        self.state["messages"].append(new_msg)
        
        # Invoke the graph
        try:
            # We pass the current state to the graph. The graph reducer `add_messages` 
            # will append the AI message to the internal state if we use a checkpointer,
            # but since we're running it linearly here, we just take the output.
            result_state = self.graph.invoke(self.state)
            
            # Update local state with the graph's returned messages 
            # (which includes the AI response appended to the history)
            self.state["messages"] = result_state["messages"]
            
            # The last message is the AI response
            ai_response = result_state["messages"][-1].content
            return ai_response
            
        except Exception as e:
            return f"Error executing LangGraph: {e}"

if __name__ == "__main__":
    agent = MemoryAgent(user_id="test_user_langgraph")
    print("LangGraph Memory Agent Initialized. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        res = agent.chat(query)
        print(f"AI: {res}\n")
