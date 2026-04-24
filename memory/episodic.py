import json
import os
from datetime import datetime

EPISODES_FILE = "episodes.json"

def log_episode(event: str, tags: list[str], filepath=EPISODES_FILE):
    """
    Logs an episodic memory event.
    """
    episode = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "tags": tags
    }
    
    episodes = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                episodes = json.load(f)
        except json.JSONDecodeError:
            pass
            
    episodes.append(episode)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

def get_recent_episodes(filepath=EPISODES_FILE, limit=5) -> list[dict]:
    """
    Retrieves the most recent episodic events.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        return episodes[-limit:]
    except (json.JSONDecodeError, FileNotFoundError):
        return []
