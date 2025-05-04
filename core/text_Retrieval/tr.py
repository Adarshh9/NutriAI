import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Load FAISS index
index = faiss.read_index("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/food_disease_faiss_index.bin")

# Load metadata and texts
with open("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

with open("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Search function
def search(query, top_k=5):
    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Format results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            "text": texts[idx],
            "source": metadata[idx]["source"],
            "id": metadata[idx]["id"],
            "score": float(1 / (1 + distance))  # Similarity score
        })
    return results

# Example usage
query = "foods that help in asthma prevention"
results = search(query)

# Show results
for result in results:
    print(f"Source: {result['source']}")
    print(f"PMID: {result['id']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...\n")

