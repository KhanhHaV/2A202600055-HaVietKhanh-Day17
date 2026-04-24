import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

try:
    client = chromadb.Client()
    
    # Use Chroma's built-in default embedding function (all-MiniLM-L6-v2)
    # This provides real semantic similarity instead of mock constant vectors
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="semantic_memory",
        embedding_function=ef
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChromaDB: {e}")
    collection = None

def store_semantic(doc_id: str, text: str, metadata: dict = None):
    """
    Stores a semantic memory document into ChromaDB with real embeddings.
    """
    if not collection:
        return
    if metadata is None:
        metadata = {}
    collection.upsert(documents=[text], ids=[doc_id], metadatas=[metadata])

def query_semantic(query: str, n_results: int = 3):
    """
    Queries semantic memory for the most relevant documents using vector similarity.
    """
    if not collection:
        return {"documents": [[]]}
    # Avoid querying if collection is empty
    if collection.count() == 0:
         return {"documents": [[]]}
         
    # Adjust n_results if collection size is smaller than requested
    n = min(n_results, collection.count())
    if n == 0:
        return {"documents": [[]]}
        
    return collection.query(query_texts=[query], n_results=n)
