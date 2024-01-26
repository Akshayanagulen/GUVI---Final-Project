import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk import download

import string
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np

# Download NLTK data and VADER lexicon (if not already downloaded)
download('all')
download('stopwords')
download('punkt')
download('wordnet')
download('maxent_ne_chunker')
download('words')
download('vader_lexicon')

# Import streamlit
import streamlit as st

#streamlit code

# Title
st.title("Text analysis using NLP")
# Header
st.header("Sentimental Analysis using NLP") 

def get_input(prompt):
    nlp_text = input(prompt)
    return nlp_text

user_input = st.text_input("Enter the text for NLP Analysis : ")
st.write(user_input)

st.subheader("NLP preprocessing Data")
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

preprocessed_text = preprocess_text(user_input)

st.write("Preprocessed Text:")

st.text(" ".join(preprocessed_text))

st.subheader("Keywords")

def get_frequency(preprocessed_text):
    # Calculate word frequencies
    word_frequencies = Counter(preprocessed_text)

    # Display the most common words as keywords
    keywords = word_frequencies.most_common(5)  # Adjust the number as needed
    
    return keywords

keywords = get_frequency(preprocessed_text)
for keyword, frequency in keywords:
    st.write(f"{keyword}: {frequency} times")

st.subheader("NER Function using NLP")
st.write("Named Entity Recognition (NER) involves identifying and classifying named entities (such as names of people, organizations, locations, dates, etc.) ")
# Function to perform NER on user input
def ner_user_input(user_input):
    # Tokenize the input
    tokens = word_tokenize(user_input)

    # Part-of-speech tagging
    pos_tags = pos_tag(tokens)

    # Perform Named Entity Recognition
    ner_tree = ne_chunk(pos_tags)

    # Extract and display named entities
    named_entities = []
    for entity in ner_tree:
        if isinstance(entity, nltk.Tree):
            label = entity.label()
            entity_text = " ".join([word for word, tag in entity.leaves()])
            named_entities.append((entity_text, label))

    return named_entities

# Perform NER on user input
result = ner_user_input(user_input)

# Display the named entities and their labels
st.write("Named Entities:")
for entity, label in result:
    st.write(f"{entity}: {label}")

st.subheader("Sentimental Analysis")

def analyze_sentiment(user_input):
    # Initialize the Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Get the sentiment scores
    sentiment_scores = sia.polarity_scores(user_input)

    # Determine the sentiment based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Return the sentiment and the sentiment scores
    return sentiment, sentiment_scores

sentiment_result, sentiment_scores_result = analyze_sentiment(user_input)

# Print the sentiment and the sentiment scores
st.write("Sentiment Scores")
st.table([sentiment_scores_result])
st.write("Sentiment Result:", sentiment_result)

# Data for the bar chart
labels = ['Negative', 'Neutral', 'Positive']
values = [sentiment_scores_result['neg'], sentiment_scores_result['neu'], sentiment_scores_result['pos']]
# Set a smaller figure size
fig, ax = plt.subplots(figsize=(4, 2))
ax.bar(labels, values, color=['red', 'yellow', 'green'])
ax.set_title('Sentiment Analysis')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Score')

# Display the bar chart in Streamlit
st.pyplot(fig)

st.subheader("Word Cloud")

def generate_wordcloud(text_data):
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text_data)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

    # Display the WordCloud using st.image
    st.image(wordcloud.to_array(), width=400)

# Call the function with user input
generate_wordcloud(user_input)
