import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up the path to the dataset
file_path = 'C:/Users/shali/Desktop/HEX INT/Project 2/IMDB-Movie-Data.csv'  # Ensure this is the correct file name and path

# Load the dataset and handle file errors
try:
    movies_df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file path.")
    raise  # Raise the error to stop execution if the file is not found

# Inspect the DataFrame
print("Column names:", movies_df.columns.tolist())  # Print the list of column names
print(movies_df.head())

# Check for missing values
print("Missing values in each column:\n", movies_df.isnull().sum())

# Optional: Strip any leading or trailing whitespace from column names
movies_df.columns = movies_df.columns.str.strip()

# Step 5: Check if 'avg_vote' exists
if 'Votes' in movies_df.columns:
    # Calculate summary statistics
    mean_rating = movies_df['Votes'].mean()
    median_rating = movies_df['Votes'].median()
    mode_rating = movies_df['Votes'].mode()[0]

    print(f"Mean Rating: {mean_rating}")
    print(f"Median Rating: {median_rating}")
    print(f"Mode Rating: {mode_rating}")

    # Visualize the distribution of ratings
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(movies_df['Votes'], bins=20, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Average Vote')
    plt.ylabel('Frequency')

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=movies_df['Votes'])
    plt.title('Box Plot of Ratings')
    plt.xlabel('Average Vote')

    plt.tight_layout()
    plt.show()

    # Identify top-rated movies
    top_rated_movies = movies_df.nlargest(10, 'Votes')
    print("Top Rated Movies:")
    print(top_rated_movies[['Title', 'Votes']])  # Assuming 'title' is the column with movie names

    # Identify top genres (assuming there is a 'genre' column)
    if 'genre' in movies_df.columns:
        top_genres = movies_df['genre'].value_counts().head(10)
        print("Top Genres:")
        print(top_genres)

        # Visualize top genres
        #barplot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_genres.index, y=top_genres.values)
        plt.title('Top Genres')
        plt.xlabel('Genre')
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("'genre' column not found. Please check the column names.")
else:
    print("'Votes' column not found. Please check the column names.")