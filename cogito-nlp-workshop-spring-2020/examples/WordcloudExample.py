from utils import DatasetLoader
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# This exmample shows how to display a word cloud generated from the verbs in the reviews.

# Load dataset
dataset = DatasetLoader.load_reviews()

# A list that will eventually contain all adjectives from all the reviews
adjectives = []
for row in dataset:
        for word, tag in row.tagged_words():
            # POS-tags starting with "VB" refer to verbs.
            if tag.startswith("VB"):
                adjectives.append(word)

# The WordCloud-library takes in a string, so we need to concatenate all adjectives in order to have one large string.
adjective_string = " ".join(adjectives)

# Configure word cloud plot
wordcloud = WordCloud(width=1000, height=500).generate(adjective_string)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
