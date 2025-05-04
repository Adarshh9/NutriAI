import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Load the dataset
df = pd.read_csv("dataset/full_dataset.csv")  # Replace with your dataset path

# Step 2: Preprocess the data
# Combine food and disease entities into a single text field
df["food_disease_pair"] = df["food_entity"] + " " + df["disease_entity"]

# Create a combined label: 1 for recommend, -1 for avoid, 0 for neutral
df["label"] = df["is_treat"] - df["is_cause"]

# Step 3: Generate embeddings
# Load a pre-trained sentence embedding model (e.g., Sentence-BERT)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for food-disease pairs
embeddings = model.encode(df["food_disease_pair"].tolist())

# Step 4: Build FAISS index
# Convert embeddings to numpy array
embeddings = np.array(embeddings).astype("float32")

# Initialize FAISS index
dimension = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Add embeddings to the index
index.add(embeddings)

# Step 5: Save the FAISS index and preprocessed data
faiss.write_index(index, "food_disease_index.faiss")
df.to_csv("preprocessed_data.csv", index=False)

print("Training complete! FAISS index and preprocessed data saved.")