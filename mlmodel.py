!pip install streamlit transformers pandas plotly
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import plotly.express as px  # Use Plotly for enhanced visuals

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
    st.title("Sentiment Analysis of News Articles")
    file_path = "F:/programming/FK_it/company_news1.csv"  # Update your file path accordingly
    df = pd.read_csv(file_path)

    if 'content' not in df.columns:
        st.error("The expected 'content' column is not in the CSV.")
        return

    # Perform sentiment analysis
    st.write("Analyzing sentiment of news articles...")
    df[['sentiment', 'sentiment_numeric']] = df['content'].apply(lambda x: pd.Series(analyze_sentiment(x)))

    # Show data preview
    st.write("Preview of analyzed data:", df.head())

    # Plot sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    st.write("Sentiment Distribution")
    fig = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title="Sentiment Analysis of News Contents", 
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
