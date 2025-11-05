import os

# Avoid importing TensorFlow/Keras via transformers when loading sentence-transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import chromadb
from chromadb.utils import embedding_functions

def create_chromadb_collection(name: str = 'car_reviews'):
    """
    Create or Load a ChromaDB collection for car reviews
    """
    client = chromadb.PersistentClient(path="D:\\market_car_analysis\\chroma_db")
    # Use SentenceTransformers embeddings (no Ollama required for embeddings)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists to avoid embedding function mismatch
    try:
        client.delete_collection(name=name)
        print(f"Deleted existing collection '{name}' to recreate with correct embedding function.")
    except Exception:
        pass  # Collection doesn't exist, which is fine

    collection = client.create_collection(name=name, embedding_function=embedding_fn)

    return collection

def insert_reviews_to_chromadb(collection, data:dict, batch_size: int = 5000):
    """
    Insert car reviews data into ChromaDB collection in batches
    """
    documents = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]
    
    total = len(documents)
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_docs = documents[i:end_idx]
        batch_metas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
        )
        print(f"Inserted batch {i//batch_size + 1}: {end_idx - i} documents (total: {end_idx}/{total})")
    
    print(f"Successfully inserted all {total} documents into ChromaDB.")

def query_reviews_from_chromadb(collection, query:str, n_results:int = 5):
    """
    Query car reviews from ChromaDB collection
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    return results
