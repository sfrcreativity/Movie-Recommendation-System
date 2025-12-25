import streamlit as st
import requests
import joblib
from dotenv import load_dotenv
import os

# Load data and models
movies = joblib.load("movies.pkl")  # Ensure CSV has 'title' and 'id'
similarity = joblib.load("similarity_matrix.pkl")  # Or recompute if needed

load_dotenv()  # Load environment variables from .env
API_KEY = os.getenv("API_KEY")
#API_KEY = st.secrets["TMDB"]["API_KEY"]

# Fetch poster for a given movie_id from TMDb
def fetch_poster(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        )
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        return None
    except Exception:
        st.warning("Could not fetch poster for this movie.")
        return None
    
# Recommendation function
def recommend(movie_title, top_n=10):
    # Find the index of the movie based on its title
    index = movies[movies["title"] == movie_title].index[0]
    
    # Get the similarity scores for the given movie
    distances = similarity[index]

    # Sort the movies by similarity score, ignoring the input movie itself
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:top_n+1]  # Skip the first element because it's the movie itself

    # Return a list of tuples: (movie_title, movie_id as int)
    return [(fetch_poster(movies.iloc[i[0]].id), movies.iloc[i[0]].title) for i in movie_list]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)

    if recommendations:
        # Create 3 columns
        cols = st.columns(3)
        for i, (poster, title) in enumerate(recommendations):
            if poster:
                with cols[i % 3]:  # Distribute movies across 3 columns
                    st.image(poster, width=150)
                    st.write(title)

# 5. Add a sidebar with info
st.sidebar.info("This app uses tf-idf vectorization and cosine similarity to recommend movies based on your selection. It fetches movie posters using The Movie Database (TMDb) API.")

st.sidebar.markdown("""
### About
This Movie Recommendation System suggests movies similar to the one you select, based on content features.
### Instructions
1. Select a movie from the dropdown.
2. Click the "Show Recommendations" button to see similar movies.
""")
st.markdown("---")
st.markdown("Developed by Syed Fazlur Rehman | [GitHub Repository](https://github.com/sfrcreativity/Movie-Recommendation-System.git)")