import tkinter as tk
from tkinter import ttk, scrolledtext
from book_recommender import BookRecommender, clean_data
from visualizer import plot_most_rated_books, plot_reviewer_behavior, plot_rating_count_vs_average
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import logging

class BookRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Recommendation System")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar - Create this FIRST
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.status_var.set("Initializing...")
        
        # Initialize the recommender
        self.initialize_recommender()
        
        # Search section
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(search_frame, text="Enter Book Title:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Get Recommendations", 
                  command=self.show_recommendations).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Random Books", 
                  command=self.show_random_books).pack(side=tk.LEFT)
        
        # Results area
        ttk.Label(main_frame, text="Recommendations:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        self.results_area = scrolledtext.ScrolledText(main_frame, width=80, height=10)
        self.results_area.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Visualization buttons
        viz_frame = ttk.LabelFrame(main_frame, text="Visualizations", padding="5")
        viz_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Button(viz_frame, text="Most Rated Books", 
                  command=lambda: self.show_visualization('most_rated')).grid(row=0, column=0, padx=5)
        ttk.Button(viz_frame, text="Reviewer Behavior", 
                  command=lambda: self.show_visualization('reviewer')).grid(row=0, column=1, padx=5)
        ttk.Button(viz_frame, text="Rating Analysis", 
                  command=lambda: self.show_visualization('regression')).grid(row=0, column=2, padx=5)
        
        # Visualization area
        self.viz_frame = ttk.Frame(main_frame)
        self.viz_frame.grid(row=4, column=0, columnspan=3, pady=10)
        self.current_canvas = None

    def initialize_recommender(self):
        clean_data()  # Clean data if needed
        
        if os.path.exists('book_recommender_model.pkl'):
            self.status_var.set("Loading existing model...")
            try:
                self.recommender = BookRecommender.load_model()
                self.status_var.set("Model loaded successfully!")
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                return
        else:
            self.status_var.set("Training new model...")
            try:
                self.recommender = BookRecommender(n_factors=50)
                ratings_matrix = self.recommender.load_and_prepare_data('Cleaned_data.csv')
                self.status_var.set("Analyzing model accuracy...")
                self.recommender.analyze_accuracy(ratings_matrix)
                self.status_var.set("Training final model...")
                self.recommender.train(ratings_matrix)
                self.recommender.save_model()
                self.status_var.set("Model trained and saved successfully!")
            except Exception as e:
                self.status_var.set(f"Error during model training: {str(e)}")
                return

    def show_recommendations(self):
        book_title = self.search_entry.get()
        if not book_title:
            self.status_var.set("Please enter a book title")
            return
            
        result = self.recommender.get_recommendations(book_title)
        self.results_area.delete(1.0, tk.END)
        
        if result is None:
            self.results_area.insert(tk.END, "No recommendations found.")
        elif 'recommendations' in result:
            for i, book in enumerate(result['recommendations'], 1):
                rec_text = f"{i}. {book['title']} by {book['author']}"
                rec_text += f" (Similarity: {book['similarity']:.4f})\n\n"
                self.results_area.insert(tk.END, rec_text)
        else:
            self.results_area.insert(tk.END, "An error occurred while retrieving recommendations.")
            
    def show_random_books(self):
        books = self.recommender.get_random_books(10)
        self.results_area.delete(1.0, tk.END)
        if books:
            self.results_area.insert(tk.END, "Random Book Selection:\n\n")
            for i, book in enumerate(books, 1):
                self.results_area.insert(tk.END, f"{i}. {book['title']} by {book['author']}\n\n")
        else:
            self.results_area.insert(tk.END, "Error retrieving random books.")

    def clear_visualization(self):
        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        plt.close('all')

    def show_visualization(self, viz_type):
        self.clear_visualization()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        if viz_type == 'most_rated':
            self.status_var.set("Generating most rated books visualization...")
            plot_most_rated_books(fig)
        elif viz_type == 'reviewer':
            self.status_var.set("Generating reviewer behavior visualization...")
            plot_reviewer_behavior(fig)
        else:  # regression
            self.status_var.set("Generating rating analysis...")
            plot_rating_count_vs_average(fig)
        
        # Create canvas and display the plot
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        
        self.status_var.set("Visualization displayed")

def main():
    root = tk.Tk()
    app = BookRecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()