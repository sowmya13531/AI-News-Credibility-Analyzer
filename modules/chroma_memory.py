import chromadb
from sentence_transformers import SentenceTransformer

# ------------------ Initialize ChromaDB ------------------
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="news_memory"
)

# ------------------ Embedding Model ------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------ Store Memory ------------------
def store_memory(text, verdict, score, explanation):
    """
    Stores analyzed news article into memory
    """

    embedding = embedder.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "verdict": verdict,
            "score": score,
            "explanation": explanation
        }],
        ids=[str(abs(hash(text)))]
    )


# ------------------ Retrieve Similar Past Articles ------------------
def retrieve_similar(text, k=3):
    """
    Retrieves similar past cases for contextual reasoning
    """

    embedding = embedder.encode(text).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    memories = []

    if results and "metadatas" in results:
        for meta in results["metadatas"][0]:
            memories.append(meta)

    return memories
