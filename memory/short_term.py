class SimpleBufferMemory:
    """
    Short-term memory with sliding window support.
    Keeps only the most recent `max_turns` exchanges to prevent unbounded growth.
    """
    def __init__(self, max_turns: int = 10):
        self.chat_history = []
        self.max_turns = max_turns

    def save_context(self, inputs: dict, outputs: dict):
        self.chat_history.append({"input": inputs.get("input", ""), "output": outputs.get("output", "")})
        # Sliding window: trim oldest turns when exceeding max
        if len(self.chat_history) > self.max_turns:
            self.chat_history = self.chat_history[-self.max_turns:]

    def load_memory_variables(self, inputs: dict) -> dict:
        return {"chat_history": self.chat_history}

    def clear(self):
        """Clears all short-term memory."""
        self.chat_history = []

def get_short_term_memory(max_turns: int = 10):
    """
    Returns a short-term memory instance with sliding window.
    """
    return SimpleBufferMemory(max_turns=max_turns)
