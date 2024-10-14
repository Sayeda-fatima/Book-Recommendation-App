import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Initialize session state for FAISS index
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

if 'combined_embeddings' not in st.session_state:
    st.session_state.combined_embeddings = None

# Load the dataset
df = pd.read_csv("books_info.csv")
st.write(f"Loaded {len(df)} books from the dataset.")

# Load the Sentence-BERT model to generate embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Combine summary, reviews, and book tropes into a single text string for each book
df['combined_text'] = df.apply(
    lambda row: f"{row['summary']} [SEP] {row['reviews']} [SEP] Tropes: {row['book_tropes']}",
    axis=1
)

# Initialize and populate FAISS index if not already done
def initialize_faiss_index():
    # Generate embeddings for the combined text
    combined_embeddings = embedder.encode(df['combined_text'].tolist())
    
    # Create and populate the FAISS index
    dimension = combined_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(combined_embeddings))

    # Store index and embeddings in session state
    st.session_state.faiss_index = index
    st.session_state.combined_embeddings = combined_embeddings

if st.session_state.faiss_index is None:
    initialize_faiss_index()

# Function to get top book recommendations based on user query
def get_recommendations(user_query, top_k=3, embedder=embedder):
    query_embedding = embedder.encode([user_query])
    distances, indices = st.session_state.faiss_index.search(query_embedding, top_k)

    recommendations = []
    for i in indices[0]:
        book = df.iloc[i]
        recommendations.append({
            "title": book['title'],
            "author": book['author'],
            "rating": book['rating'],
            "book_tropes": book['book_tropes'],
            "summary": book['summary']
        })
    return recommendations

# Streamlit UI
st.title("Tropes-Based Book Recommendation System")
st.write("Enter a trope and get book recommendations!")

# User input
user_query = st.text_input("Enter a trope (e.g., enemies-to-lovers, found family):")

if st.button("Recommend Books"):
    if user_query:
        # Get recommendations
        st.write("Finding recommendations...")
        recommendations = get_recommendations(user_query)

        # Display recommendations
        st.write("### Top Recommendations:")
        for book in recommendations:
            st.write(f"**{book['title']}** by {book['author']} (Rating: {book['rating']})")
            st.write(f"Tropes: {book['book_tropes']}")
            st.write(f"Summary: {book['summary']}")
            st.write("---")
    else:
        st.write("Please enter a trope to get recommendations.")
