import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of a single text and return the sentiment label and numeric value."""
    # Preprocess text
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())

    # Map scores to sentiment labels and numeric values
    sentiments = ["negative", "neutral", "positive"]
    sentiment = sentiments[scores.argmax()]
    
    # Convert sentiment to numeric
    sentiment_map = {
        "negative": -1,
        "neutral": 0,
        "positive": 1
    }
    
    return sentiment, sentiment_map[sentiment]

def main():
    file_path = "F:\programming\FK_it\company_news1.csv"
    df = pd.read_csv(file_path)

    # Perform sentiment analysis on the "content" column
    if 'content' not in df.columns:
        raise ValueError("The expected 'content' column is not in the CSV.")

    # Apply sentiment analysis
    df[['sentiment', 'sentiment_numeric']] = df['content'].apply(lambda x: pd.Series(analyze_sentiment(x)))

    # Pie Chart
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=["#ff9999", "#66b3ff", "#99ff99"]
    )
    plt.title("Sentiment Analysis of News Contents")
    plt.show()


if __name__ == "__main__":
    main()
