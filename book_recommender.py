import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pickle
import os
import logging
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import psutil
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_from_dropbox(url, local_filename):
    """Download a file from Dropbox if it doesn't exist locally."""
    if not os.path.exists(local_filename):
        logger.info(f"Downloading {local_filename} from Dropbox...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(local_filename, 'wb') as f:
                f.write(response.content)
            logger.info(f"Download completed: {local_filename}")
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise

def clean_data(input_file='Preprocessed_data.csv', output_file='Cleaned_data.csv'):
    """Clean the data by removing zero ratings and save to new file."""
    if os.path.exists(output_file):
        logger.info(f"Cleaned data file {output_file} already exists")
        return
    
    # Dropbox link for the data
    dropbox_url = "https://www.dropbox.com/scl/fi/xl3bewhqufmw60002orv1/Cleaned_data.csv?rlkey=2ockjpkkhhmd6e8nq6t2xg4ag&st=xcbhhd6y&dl=1"
    
    # Download the file if needed
    download_from_dropbox(dropbox_url, input_file)
    
    logger.info("Starting data cleaning")
    df = pd.read_csv(input_file)
    logger.info(f"Original data shape: {df.shape}")
    
    # Remove zero ratings
    df = df[df['rating'] > 0]
    logger.info(f"Shape after removing zero ratings: {df.shape}")
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    logger.info(f"Cleaned data saved to {output_file}")

class BookRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.model = None
        self.book_data = None
        self.user_ratings_mean = None
        self.book_ids = None
        self.user_ids = None
        self.metrics = {}

        self.min_user_ratings = 10
        self.min_book_ratings = 5
        self.max_users = 125000
        self.max_books = 100000

    def memory_status(self):
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Memory usage: {memory_gb:.2f} GB")

    def load_and_prepare_data(self, file_path):
        logger.info("Starting data loading and preparation")
        self.memory_status()

        try:
            # Load the cleaned data
            df_ratings = pd.read_csv(file_path, usecols=['user_id', 'isbn', 'rating'],
                                   dtype={'user_id': np.int32, 'isbn': str, 'rating': np.float32})

            # Calculate user and book counts
            user_counts = df_ratings['user_id'].value_counts()
            book_counts = df_ratings['isbn'].value_counts()

            valid_users = user_counts[user_counts >= self.min_user_ratings].head(self.max_users).index
            valid_books = book_counts[book_counts >= self.min_book_ratings].head(self.max_books).index
            
            df_ratings = df_ratings[
                df_ratings['user_id'].isin(valid_users) &
                df_ratings['isbn'].isin(valid_books)
            ]

            self.user_ids = {id_: idx for idx, id_ in enumerate(valid_users)}
            self.book_ids = {isbn: idx for idx, isbn in enumerate(valid_books)}

            df_books = pd.read_csv(file_path, usecols=['isbn', 'book_title', 'book_author'])
            df_books = df_books[df_books['isbn'].isin(valid_books)]
            
            self.book_data = df_books.drop_duplicates(subset=['isbn']).copy()
            self.book_data.set_index('isbn', inplace=True)
            
            ratings_matrix = csr_matrix(
                (df_ratings['rating'].values,
                 (df_ratings['user_id'].map(self.user_ids),
                  df_ratings['isbn'].map(self.book_ids))),
                shape=(len(self.user_ids), len(self.book_ids))
            )

            del df_ratings, df_books
            gc.collect()
            
            logger.info(f"Final matrix shape: {ratings_matrix.shape}")
            self.memory_status()
            
            return ratings_matrix

        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train(self, ratings_matrix):
        logger.info("Training model")
        self.memory_status()

        try:
            self.user_ratings_mean = np.array(ratings_matrix.mean(axis=1)).flatten()
            ratings_centered = ratings_matrix.copy()
            for i in range(ratings_matrix.shape[0]):
                ratings_centered.data[ratings_centered.indptr[i]:ratings_centered.indptr[i+1]] -= self.user_ratings_mean[i]

            U, sigma, Vt = svds(ratings_centered, k=self.n_factors)
            self.model = {'U': U, 'sigma': sigma, 'Vt': Vt}
            
            logger.info("Model training completed")
            self.memory_status()

        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def analyze_accuracy(self, ratings_matrix, test_size=0.2):
        logger.info("Starting accuracy analysis")
        self.memory_status()

        try:
            rows, cols = ratings_matrix.nonzero()
            n_nonzero = len(rows)
            test_size = min(int(n_nonzero * test_size), 5000)
            test_indices = np.random.choice(n_nonzero, test_size, replace=False)
            
            test_rows = rows[test_indices]
            test_cols = cols[test_indices]
            test_ratings = np.array(ratings_matrix[test_rows, test_cols]).flatten()
            
            train_matrix = ratings_matrix.copy()
            train_matrix[test_rows, test_cols] = 0
            
            self.train(train_matrix)
            
            pred_ratings = []
            for user, item in zip(test_rows, test_cols):
                pred = (
                    self.model['U'][user, :] @
                    np.diag(self.model['sigma']) @
                    self.model['Vt'][:, item] +
                    self.user_ratings_mean[user]
                )
                pred_ratings.append(pred)
            
            self.metrics = {
                'rmse': np.sqrt(mean_squared_error(test_ratings, pred_ratings)),
                'mae': mean_absolute_error(test_ratings, pred_ratings),
                'test_samples': test_size
            }
            
            logger.info(f"Accuracy Results:")
            logger.info(f"RMSE: {self.metrics['rmse']:.4f}")
            logger.info(f"MAE: {self.metrics['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in accuracy analysis: {str(e)}")
            raise

    def save_model(self, filepath='book_recommender_model.pkl'):
        logger.info("Saving model")
        try:
            model_data = {
                'model': self.model,
                'user_ratings_mean': self.user_ratings_mean,
                'book_ids': self.book_ids,
                'user_ids': self.user_ids,
                'book_data': self.book_data,
                'metrics': self.metrics,
                'n_factors': self.n_factors
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath='book_recommender_model.pkl'):
        logger.info("Loading model")
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            recommender = cls(n_factors=model_data['n_factors'])
            recommender.model = model_data['model']
            recommender.user_ratings_mean = model_data['user_ratings_mean']
            recommender.book_ids = model_data['book_ids']
            recommender.user_ids = model_data['user_ids']
            recommender.book_data = model_data['book_data']
            recommender.metrics = model_data['metrics']
            
            logger.info("Model loaded successfully")
            return recommender
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_recommendations(self, book_title, n_recommendations=5):
        logger.info(f"Getting recommendations for: {book_title}")
        try:
            book_title_lower = book_title.lower()
            matching_books = self.book_data[
                self.book_data['book_title'].str.lower().str.contains(book_title_lower, na=False)
            ].drop_duplicates()

            if matching_books.empty:
                logger.warning(f"No books found matching '{book_title}'")
                return None

            isbn = matching_books.index[0]
            book_idx = self.book_ids.get(isbn)

            if book_idx is None:
                logger.warning(f"Book index not found for ISBN: {isbn}")
                return None

            book_factors = self.model['Vt'][:, book_idx]
            similarities = self.model['Vt'].T.dot(book_factors)

            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations + 1]

            recommendations = []
            for idx in similar_indices:
                if idx == book_idx:
                    continue
                similar_isbn = next((k for k, v in self.book_ids.items() if v == idx), None)
                if similar_isbn and similar_isbn in self.book_data.index:
                    book_info = self.book_data.loc[similar_isbn]
                    recommendations.append({
                        'title': book_info['book_title'],
                        'author': book_info['book_author'],
                        'similarity': float(similarities[idx])
                    })

            return {'recommendations': recommendations}

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return None

    def get_random_books(self, n=10):
        """Get a random sample of books analyzed by the model."""
        try:
            sample_size = min(n, len(self.book_data))
            if sample_size == 0:
                logger.warning("No books available in the dataset.")
                return []

            sample = self.book_data.sample(sample_size)
            return [
                {'title': row['book_title'], 'author': row['book_author']}
                for _, row in sample.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error getting random books: {str(e)}")
            return []