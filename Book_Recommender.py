# app.py
import streamlit as st
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- GNN Model Definition ---
# This class defines the structure of our Graph Neural Network.
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        # Define three graph convolutional layers.
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass data through the layers with ReLU activation.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

# --- Data Loading and Model Preparation ---
# Use st.cache_resource to load data and initialize the model only once.
@st.cache_resource
def load_data_and_model():
    # Load the dataset from the local CSV file.
    # IMPORTANT: 'books.csv' must be in the same folder as this script.
    books_df = pd.read_csv('books.csv')
    
    # --- Data Preprocessing ---
    titles = books_df['title'].values
    authors = books_df['author'].values
    genres = books_df['genre'].values
    user_ratings = books_df['user_rating'].values

    title_encoder = LabelEncoder()
    author_encoder = LabelEncoder()
    genre_encoder = LabelEncoder()

    titles_encoded = title_encoder.fit_transform(titles)
    # Create a mapping from title to its encoded ID for safer lookups
    title_to_id = {title: i for i, title in enumerate(title_encoder.classes_)}

    authors_encoded = author_encoder.fit_transform(authors)
    genres_encoded = genre_encoder.fit_transform(genres)

    scaler = MinMaxScaler()
    user_ratings_normalized = scaler.fit_transform(user_ratings.reshape(-1, 1)).squeeze()

    node_features = torch.tensor(
        list(zip(titles_encoded, authors_encoded, genres_encoded, user_ratings_normalized)),
        dtype=torch.float
    )

    # A simple edge creation strategy: connect consecutive books
    edge_index = torch.tensor([[i, i + 1] for i in range(len(titles) - 1)], dtype=torch.long).t().contiguous()
    data = Data(x=node_features, edge_index=edge_index)
    
    # --- Model Initialization ---
    num_classes = len(titles)
    model = GNN(data.num_node_features, num_classes)
    
    # In a real-world scenario, you would load pre-trained model weights here.
    # For this example, we are using the initialized model.
    
    return model, data, title_encoder, books_df, title_to_id

# --- Recommendation Function ---
def recommend(book_title, model, data, title_encoder, title_to_id, top_k=5):
    try:
        # Find the closest match for the user's input title
        all_titles = title_encoder.classes_
        # Simple case-insensitive match
        matching_titles = [t for t in all_titles if book_title.lower() in t.lower()]
        
        if not matching_titles:
            return None, "No matching book found. Please try another title."

        # Use the first match
        matched_title = matching_titles[0]
        # Use the pre-computed dictionary for a safe lookup instead of transform
        title_encoded = title_to_id[matched_title]

    except (ValueError, KeyError):
        return None, "Could not process the book title."

    model.eval()
    with torch.no_grad():
        out = model(data)

    scores = out[title_encoded]
    _, indices = torch.topk(scores, top_k + 1)
    
    recommendations_indices = indices.cpu().numpy()
    recommendations_indices = [idx for idx in recommendations_indices if idx != title_encoded]
    
    recommendations = title_encoder.inverse_transform(recommendations_indices[:top_k])
    return recommendations, f"Recommendations based on '{matched_title}':"

# --- Streamlit App UI ---
st.set_page_config(page_title="Book Recommender", layout="wide", page_icon="�")

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.info("This app uses a Graph Neural Network (GNN) to recommend books based on title, author, genre, and user ratings.")
    st.markdown("---")
    st.header("Options")
    top_k = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5, key="rec_slider")
    st.markdown("---")
    st.markdown("Created by **Chitransh Shukla**")

# --- Main Page ---
st.title("📖 Modern Book Recommender")
st.markdown("Discover your next favorite book. Enter a title you enjoy, and get personalized recommendations.")

try:
    # Load everything once at the start
    model, data, title_encoder, books_df, title_to_id = load_data_and_model()

    # --- Callback functions to clear the other input ---
    def clear_text_input():
        st.session_state.text_input = ""

    def clear_selectbox():
        # Set selectbox to the first item, which is the empty placeholder
        st.session_state.selectbox = book_list[0]

    # --- User Input Section in a container ---
    with st.container():
        st.header("🔍 Find Your Next Read")
        
        book_list = [""] + sorted(books_df['title'].unique().tolist())
        
        # Dropdown for selecting a book
        selected_book_dropdown = st.selectbox(
            "Select a book from the list:", 
            options=book_list, 
            key="selectbox", 
            on_change=clear_text_input
        )

        # Text input for the user to enter a book title
        user_input_title = st.text_input(
            "Or type to search for a book title:", 
            placeholder="e.g., The Great Gatsby", 
            key="text_input", 
            on_change=clear_selectbox
        )
        
        if st.button("✨ Get Recommendations"):
            # Prioritize the text input. If it's empty, use the dropdown selection.
            book_to_recommend = st.session_state.text_input if st.session_state.text_input else st.session_state.selectbox

            if book_to_recommend:
                with st.spinner('Curating recommendations for you...'):
                    recommendations, message = recommend(
                        book_to_recommend,
                        model,
                        data,
                        title_encoder,
                        title_to_id,
                        top_k=top_k
                    )
                    
                    st.success(message)
                    if recommendations is not None:
                        st.subheader(f"Here are {len(recommendations)} books you might like:")
                        for i, book in enumerate(recommendations):
                            st.markdown(f"**{i+1}.** {book}")
            else:
                st.warning("Please select or enter a book title.")

except FileNotFoundError:
    st.error("Error: 'books.csv' not found.")
    st.info("Please make sure the 'books.csv' file is in the same directory as the 'app.py' script.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")