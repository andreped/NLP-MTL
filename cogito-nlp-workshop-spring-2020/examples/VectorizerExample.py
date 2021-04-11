import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import DatasetLoader

# This example finds the most "similar" review (in terms of words used) based on a user-generated review. This is done
# by creating a so-called bag-of-words model. Each unique word in the dataset is given an index in a vector. Each review
# is in turn transformed into a vector, where the value in each index represents how many times a specific word is
# present. For instance, the word "dress" may have index=159. If a review has the value 3 at index 159, "dress" is
# mentioned 3 times in the review. The vector representation is a format that can be taken as input by machine learning
# algorithms.
#
# In this task, we calculate the cosine similarity of two vectors, and obtain a metric on how similar they are. Note
# that this is in terms of which words are used -- the vectors have no understanding on how the different words relate.

# Load dataset
dataset = DatasetLoader.load_reviews()

# Transform each text in the dataset to its corresponding vector.
print("Vectorizing dataset")
vectorizer = TfidfVectorizer()
texts = [row.full_text() for row in dataset]
vectorized_dataset = vectorizer.fit_transform(texts)


# Method for finding the review in the dataset most similar to `query`.
def find_most_similar_review(query: str) -> str:
    # find the vector of the query
    vectorized_query = vectorizer.transform([query])

    # Transform all reviews' vectors to the cosine similarity to the vector of query. This is a measurement on how
    # similar the vectors are.
    cosine_similarities = cosine_similarity(vectorized_query, vectorized_dataset).flatten()

    # Find the index of the highest cosine similarity, and return the text that exist on this index.
    most_similar_index = np.argmax(cosine_similarities)
    return texts[most_similar_index]


while True:
    # Get user input and print most similar review.
    query = input("What is your review?")
    most_similar_review = find_most_similar_review(query)
    print("\nMost similar review:\n")
    print(most_similar_review)
