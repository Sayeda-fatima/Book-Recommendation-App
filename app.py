import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the dataset
df = pd.read_csv("books_info.csv")
st.write(f"Loaded {len(df)} books from the dataset.")

# Load the Sentence-BERT model to generate embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all book summaries
summary_embeddings = embedder.encode(df['summary'].tolist())

# Create and populate the FAISS index
dimension = summary_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(summary_embeddings))

# Load LLaMA model and tokenizer for explanation generation
llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Function to get top book recommendations based on user query
def get_recommendations(user_query, top_k=3, embedder=embedder):
    query_embedding = embedder.encode([user_query])
    distances, indices = index.search(query_embedding, top_k)

    recommendations = []
    for i in indices[0]:
        book = df.iloc[i]
        recommendations.append({
            "title": book['title'],
            "author": book['author'],
            "rating": book['rating'],
            "tropes": book['book_tropes'],
            "summary": book['summary']
        })

    return recommendations

# Function to generate an explanation using LLaMA
def generate_explanation(user_query, recommendations):
    prompt = f"User asked for book recommendations based on the trope '{user_query}'.\n"
    prompt += "Here are the recommended books:\n\n"

    for i, book in enumerate(recommendations):
        prompt += f"{i + 1}. {book['title']} by {book['author']} (Rating: {book['rating']})\n"
        prompt += f"Tropes: {book['tropes']}\n"

    prompt += "\nExplain why these books are suitable recommendations."

    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_new_tokens=150)

    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

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
            st.write(f"Tropes: {book['tropes']}")
            st.write(f"Summary: {book['summary']}")
            st.write("---")

        # Generate explanation
        explanation = generate_explanation(user_query, recommendations)
        st.write("### Explanation:")
        st.write(explanation)
    else:
        st.write("Please enter a trope to get recommendations.")
