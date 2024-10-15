import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the dataset and model
df = pd.read_csv("books_info.csv")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare combined text and generate embeddings
df['combined_text'] = df.apply(
    lambda row: f"{row['summary']} [SEP] {row['reviews']} [SEP] Tropes: {row['book_tropes']}",
    axis=1
)
combined_embeddings = embedder.encode(df['combined_text'].tolist())

# Create and populate FAISS index
dimension = combined_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(combined_embeddings))

# Save the index and combined embeddings to disk
faiss.write_index(index, "faiss_index.bin")
np.save("combined_embeddings.npy", combined_embeddings)
print("FAISS index and embeddings saved.")
