import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Counts the number of tokens in a text string.
    Falls back to cl100k_base if model is not recognized.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class ContextWindowManager:
    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4o-mini"):
        self.max_tokens = max_tokens
        self.model = model
        # Each slot has a priority (1 is highest/critical, 4 is lowest) and content
        self.slots = {
            "system": {"priority": 1, "content": ""},
            "recent_turns": {"priority": 2, "content": ""},
            "memory_context": {"priority": 3, "content": ""},
            "old_history": {"priority": 4, "content": ""},
        }
    
    def total_tokens(self) -> int:
        total = 0
        for slot in self.slots.values():
            if slot["content"]:
                total += count_tokens(slot["content"], self.model)
        return total
    
    def auto_trim(self):
        """
        Automatically trims content when approaching the limit.
        Evicts according to priority from lowest (P4) to highest (P1).
        """
        # Sort slots by priority descending (P4 first, then P3, P2, P1)
        sorted_slots = sorted(
            self.slots.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        while self.total_tokens() > self.max_tokens * 0.9:  # Trim when >= 90%
            trimmed_something = False
            for slot_name, slot in sorted_slots:
                if slot["priority"] == 1:
                    continue  # Never delete P1 (System prompt)
                if slot["content"]:
                    # Cut 30% of this slot's content
                    words = slot["content"].split()
                    new_length = int(len(words) * 0.7)
                    if new_length > 0:
                        slot["content"] = " ".join(words[:new_length])
                        trimmed_something = True
                    else:
                        slot["content"] = ""
                        trimmed_something = True
                    break # Re-evaluate token count after trimming one slot
            
            if not trimmed_something:
                break  # Nothing left to trim except P1
    
    def build_context(self) -> str:
        """
        Combines all slots into a context string to feed to the LLM.
        """
        self.auto_trim()
        parts = []
        for slot in sorted(self.slots.values(), key=lambda x: x["priority"]):
            if slot["content"]:
                parts.append(slot["content"])
        return "\n\n".join(parts)
