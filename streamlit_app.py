import streamlit as st
from book_recommender import BookRecommender
from visualizer import plot_most_rated_books, plot_reviewer_behavior, plot_rating_count_vs_average
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(
    page_title="Book Recommender System",
    layout="wide"
)

# Add title and description
st.title("Book Recommender System")
st.write("This system recommends books based on collaborative filtering using SVD.")

# Cache the model loading so it only loads once
@st.cache_resource
def load_recommender():
    try:
        return BookRecommender.load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the recommender model
recommender = load_recommender()

# Create sidebar menu
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Model Accuracy", "Book Recommendations", "Random Books", "Visualizations"]
)

# Main content area
if option == "Model Accuracy":
    st.header("Model Accuracy Metrics")
    if recommender and recommender.metrics:
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "RMSE (Root Mean Square Error)", 
                f"{recommender.metrics['rmse']:.4f}",
                help="Lower values indicate better prediction accuracy"
            )
        
        with col2:
            st.metric(
                "MAE (Mean Absolute Error)", 
                f"{recommender.metrics['mae']:.4f}",
                help="Lower values indicate better prediction accuracy"
            )
        
        st.write(f"These metrics are calculated based on {recommender.metrics.get('test_samples', 'N/A')} test samples")
        
        # Add explanation of metrics
        st.markdown("""
        ### Understanding the Metrics
        
        - **RMSE (Root Mean Square Error)**: Measures the standard deviation of the prediction errors. 
          A value less than 1.0 indicates good prediction accuracy.
        
        - **MAE (Mean Absolute Error)**: Measures the average absolute differences between predicted and actual ratings. 
          A value less than 1.0 indicates good prediction accuracy.
        """)

elif option == "Book Recommendations":
    st.header("Get Book Recommendations")
    book_title = st.text_input("Enter a book title:")
    
    if book_title:
        with st.spinner("Finding recommendations..."):
            result = recommender.get_recommendations(book_title)
            if result and 'recommendations' in result:
                for i, book in enumerate(result['recommendations'], 1):
                    st.write(f"{i}. {book['title']} by {book['author']} "
                           f"(Similarity: {book['similarity']:.4f})")
            else:
                st.warning("No recommendations found for this book.")

elif option == "Random Books":
    st.header("Random Book Suggestions")
    if st.button("Get Random Books"):
        with st.spinner("Finding random books..."):
            books = recommender.get_random_books(10)
            if books:
                for i, book in enumerate(books, 1):
                    st.write(f"{i}. {book['title']} by {book['author']}")
            else:
                st.error("Error retrieving random books.")

else:  # Visualizations
    st.header("Data Visualizations")
    
    viz_type = st.radio(
        "Select visualization:",
        ["Most Rated Books", "Reviewer Behavior", "Rating Analysis"]
    )
    
    if viz_type == "Most Rated Books":
        with st.spinner("Generating visualization..."):
            fig = plot_most_rated_books()
            st.pyplot(fig)
            st.write("This chart shows the top 10 books with the most ratings.")
            
    elif viz_type == "Reviewer Behavior":
        with st.spinner("Generating visualization..."):
            fig = plot_reviewer_behavior()
            st.pyplot(fig)
            st.write("This scatter plot shows how users' average ratings relate to how many books they've rated.")
            
    else:  # Rating Analysis
        with st.spinner("Generating visualization..."):
            fig = plot_rating_count_vs_average()
            st.pyplot(fig)
            st.write("This plot shows the relationship between a book's popularity and its average rating.")

# Add footer with information about the model
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
if recommender and recommender.metrics:
    st.sidebar.write(f"Number of factors: {recommender.n_factors}")
    st.sidebar.write(f"Test samples: {recommender.metrics.get('test_samples', 'N/A')}")
st.sidebar.write("Book Recommender System using SVD")