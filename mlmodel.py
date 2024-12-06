import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import News_loader
from dotenv import load_dotenv
import os

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of a single text and return the sentiment label."""
    # Preprocess text
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())

    # Map scores to sentiment labels
    sentiments = ["negative", "neutral", "positive"]
    sentiment = sentiments[scores.argmax()]
    return sentiment

def main():
    i = 1
    file_path = "F:\programming\FK_it\company_news1.csv"
    i+=1  # Ensure this matches the path from newsloader
    df = pd.read_csv(file_path)

    # Perform sentiment analysis on the "content" column
    if 'content' not in df.columns:
        raise ValueError("The expected 'article' column is not in the CSV.")

    df['sentiment'] = df['content'].apply(analyze_sentiment)

    # Count sentiment categories
    sentiment_counts = df['sentiment'].value_counts()

    # Visualize as a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=["#ff9999", "#66b3ff", "#99ff99"]
    )
    plt.title("Sentiment Analysis of News contents")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['sentiment_numeric'], marker='o', linestyle='-', color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Sentiment Analysis Over Time")
    plt.xlabel("Article Index")
    plt.ylabel("Sentiment (1 = Positive, 0 = Neutral, -1 = Negative)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
