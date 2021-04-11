from textblob import TextBlob
import matplotlib.pyplot as plt
from utils import DatasetLoader

# Each review contains a rating in the range [1, 5]. The sentiment value of a review is a measure on how
# positive/negative the review is, and is in the range [-1, 1]. That is, positive reviews will generally have high
# sentiment scores, while negative reviews generally have low sentiment scores. This example obtains the average
# sentiment for all reviews with rating=1, ..., rating=5, giving us 5 data points which are finally plotted.

# Load dataset
dataset = DatasetLoader.load_reviews()

# Create dictionary with 5 buckets -- one for each rating number.
sentiments = {1: [], 2: [], 3: [], 4: [], 5: []}

# For each review, find the sentiment and append the value to the relevant rating's bucket.
print("Calculating sentiment values")
for row in dataset:
    blob = TextBlob(row.full_text())
    sentiment = blob.sentiment.polarity
    sentiments[row.rating].append(sentiment)

# Create lists of x and y values. The y values are the average sentiment for each rating.
x = sentiments.keys()
y = [sum(sents) / len(sents) for rating, sents in sentiments.items()]

# Configure plot
print("Plotting figure")
plt.scatter(x, y)
plt.title("Average sentiment per rating")
plt.xlabel("Rating")
plt.ylabel("Average sentiment")
plt.xticks([1, 2, 3, 4, 5])
plt.show()
