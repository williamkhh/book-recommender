from book_recommender import BookRecommender, clean_data
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("\n=== Book Recommendation System ===")
    
    # Clean data if needed
    clean_data()
    
    if os.path.exists('book_recommender_model.pkl'):
        print("\nLoading existing model...")
        try:
            recommender = BookRecommender.load_model()
            print("\nModel loaded successfully!")
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            return
    else:
        print("\nTraining new model...")
        try:
            recommender = BookRecommender(n_factors=50)
            ratings_matrix = recommender.load_and_prepare_data('Cleaned_data.csv')
            print("\nAnalyzing model accuracy...")
            recommender.analyze_accuracy(ratings_matrix)
            print("\nTraining final model...")
            recommender.train(ratings_matrix)
            recommender.save_model()
            print("\nModel trained and saved successfully!")
        except Exception as e:
            print(f"\nError during model training: {str(e)}")
            return

    while True:
        print("\n=== Book Recommendation System ===")
        print("Commands:")
        print("- Enter a book title to get recommendations")
        print("- 'browse' to see random books")
        print("- 'plot' to generate most rated books visualization")
        print("- 'plot ratings' to see reviewer behavior analysis")
        print("- 'plot regression' to analyze rating count vs average rating")
        print("- 'quit' to exit")
        
        command = input("\nEnter command: ").strip().lower()
        
        if command == 'quit':
            break
        elif command == 'browse':
            print("\nRandom selection of books analyzed by the model:")
            books = recommender.get_random_books(10)
            if books:
                for i, book in enumerate(books, 1):
                    print(f"{i}. {book['title']} by {book['author']}")
            else:
                print("Error retrieving books.")
        elif command == 'plot':
            print("\nGenerating visualization of most rated books...")
            from visualizer import plot_most_rated_books
            plot_most_rated_books()
            print("Visualization saved as 'top_rated_books.png'")
        elif command == 'plot ratings':
            print("\nGenerating reviewer behavior visualization...")
            from visualizer import plot_reviewer_behavior
            plot_reviewer_behavior()
            print("Visualization saved as 'reviewer_behavior.png'")
        elif command == 'plot regression':
            print("\nGenerating rating count vs average rating analysis...")
            from visualizer import plot_rating_count_vs_average
            plot_rating_count_vs_average()
            print("Visualization saved as 'rating_regression.png'")
        else:
            result = recommender.get_recommendations(command)
            if result is None:
                print("\nNo recommendations found.")
            elif 'recommendations' in result:
                print("\nTop recommendations:")
                for i, book in enumerate(result['recommendations'], 1):
                    print(f"{i}. {book['title']} by {book['author']} " \
                          f"(Similarity: {book['similarity']:.4f})")
            else:
                print("\nAn error occurred while retrieving recommendations.")

if __name__ == "__main__":
    main()