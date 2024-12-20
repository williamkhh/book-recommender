import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_most_rated_books(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 6))
    
    df = pd.read_csv('Cleaned_data.csv')
    
    top_books = df['isbn'].value_counts().head(10)
    
    df_books = df[['isbn', 'book_title']].drop_duplicates()
    df_books = df_books.set_index('isbn')
    titles = [df_books.loc[isbn, 'book_title'] for isbn in top_books.index]
    
    ax = fig.add_subplot(111)
    ax.bar(range(len(titles)), top_books.values)
    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels([title[:30] + '...' if len(title) > 30 else title 
                        for title in titles], rotation=45, ha='right')
    ax.set_xlabel('Book Title')
    ax.set_ylabel('Number of Ratings')
    ax.set_title('Top 10 Most Rated Books')
    
    plt.tight_layout()
    
    if fig is None:
        plt.savefig('top_rated_books.png')
        plt.close()
    
    return fig

def plot_reviewer_behavior(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    
    df = pd.read_csv('Cleaned_data.csv')
    
    user_stats = df.groupby('user_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    user_stats.columns = ['user_id', 'num_ratings', 'avg_rating']
    user_stats = user_stats[user_stats['num_ratings'] <= 100]
    
    ax = fig.add_subplot(111)
    ax.scatter(user_stats['num_ratings'], user_stats['avg_rating'], alpha=0.1)
    ax.set_xlabel('Number of Books Rated (limited to ≤100)')
    ax.set_ylabel('Average Rating')
    ax.set_title('Reviewer Behavior: Rating Count vs Average Rating')
    
    plt.tight_layout()
    
    if fig is None:
        plt.savefig('reviewer_behavior.png')
        plt.close()
    
    return fig

def plot_rating_count_vs_average(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    
    df = pd.read_csv('Cleaned_data.csv')
    
    book_stats = df.groupby('isbn').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    book_stats.columns = ['isbn', 'num_ratings', 'avg_rating']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        book_stats['num_ratings'], 
        book_stats['avg_rating']
    )
    
    ax = fig.add_subplot(111)
    ax.scatter(book_stats['num_ratings'], book_stats['avg_rating'], alpha=0.1)
    
    x_range = np.array([book_stats['num_ratings'].min(), book_stats['num_ratings'].max()])
    ax.plot(x_range, slope * x_range + intercept, color='red', 
            label=f'R² = {r_value**2:.3f}\ny = {slope:.3f}x + {intercept:.3f}')
    
    ax.set_xlabel('Number of Ratings per Book')
    ax.set_ylabel('Average Rating')
    ax.set_title('Book Popularity vs Average Rating')
    ax.legend()
    
    plt.tight_layout()
    
    if fig is None:
        plt.savefig('rating_regression.png')
        plt.close()
        print(f"\nRegression Statistics:")
        print(f"Slope: {slope:.3f}")
        print(f"Intercept: {intercept:.3f}")
        print(f"R-squared: {r_value**2:.3f}")
        print(f"P-value: {p_value:.3e}")
    
    return fig

if __name__ == "__main__":
    plot_most_rated_books()
    plot_reviewer_behavior()
    plot_rating_count_vs_average()