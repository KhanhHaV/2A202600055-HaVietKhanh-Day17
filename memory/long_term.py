import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Fallback in-memory dictionary if Redis fails
_mock_redis = {}

try:
    r = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=REDIS_DB, 
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0
    )
    r.ping() # Check if connection is actually alive
except Exception as e:
    print(f"Warning: Failed to connect to Redis ({e}). Using local mock dictionary.")
    r = None

def save_long_term(user_id: str, key: str, value: str):
    """
    Saves a user preference to Redis or local mock.
    """
    try:
        if r:
            data = r.get(f"user:{user_id}")
            memory = json.loads(data) if data else {}
            memory[key] = value
            r.set(f"user:{user_id}", json.dumps(memory))
        else:
            uid = f"user:{user_id}"
            if uid not in _mock_redis:
                _mock_redis[uid] = "{}"
            memory = json.loads(_mock_redis[uid])
            memory[key] = value
            _mock_redis[uid] = json.dumps(memory)
    except Exception as e:
        print(f"Long-term save error: {e}")

def load_long_term(user_id: str) -> dict:
    """
    Loads user preferences from Redis or local mock.
    """
    try:
        if r:
            data = r.get(f"user:{user_id}")
            return json.loads(data) if data else {}
        else:
            data = _mock_redis.get(f"user:{user_id}")
            return json.loads(data) if data else {}
    except Exception as e:
        print(f"Long-term load error: {e}")
        return {}

