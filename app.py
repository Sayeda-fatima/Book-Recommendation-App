import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load the dataset
df = pd.read_csv("books_info.csv")
# Loading Image using PIL
im = Image.open('public/icon.jpg')
# Adding Image to web app
st.set_page_config(page_title="Trope-Based Book Recommendation System", page_icon = im)
st.write(f"Loaded {len(df)} books from the dataset.")

# Load precomputed FAISS index and embeddings
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("public/faiss_index.bin")
    embeddings = np.load("public/combined_embeddings.npy")
    return index, embeddings

faiss_index, combined_embeddings = load_faiss_index()

# Load the Sentence-BERT model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get book recommendations
def get_recommendations(user_query, top_k=5):
    query_embedding = embedder.encode([user_query])
    distances, indices = faiss_index.search(query_embedding, top_k)

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
# Remove default main menu options
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
st.title("Trope-Based Book Recommendation System")
st.write("Enter a trope and get book recommendations!")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF60A7;
        color: white;
        border: none;
        padding: 10px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFACD1;  /* Lighter Pink on Hover */
    }
    .st-bc{
            background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# User input
user_query = st.text_input("Enter a trope (e.g., enemies-to-lovers, found family):")

if st.button("Recommend Books"):
    if user_query:
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
        st.write("Please enter a trope to get recommendations!")
