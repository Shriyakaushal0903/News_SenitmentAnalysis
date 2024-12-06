import requests
import pandas as pd
from datetime import datetime, timedelta
import os 
from dotenv import load_dotenv

# Your News API key
# API_KEY = os.getenv("NEWS_API_KEY")
API_KEY = '79e3842815ea48f9ae7cb26293aeb48c'

# API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise ValueError("API key not found! Make sure it's set in the .env file.")

# Function to fetch news articles
def fetch_news(company_name, api_key):
    # Base URL for News API
    url = 'https://newsapi.org/v2/everything'

    # Define the time range: past 10 days
    to_date = datetime.now()
    from_date = to_date - timedelta(days=10)

    # Parameters for the API request
    params = {
        'q': company_name,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',  # Sort by relevance
        'apiKey': api_key,
        'language': 'en',  # Only fetch English articles
    }

    # Send GET request to News API
    response = requests.get(url, params=params)
    
    # Raise exception if request fails
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles']
        return articles
    else:
        print("Error fetching articles:", data)
        return []

# Function to save articles in a DataFrame
def save_to_dataframe(articles):
    # Extract relevant fields
    data = {
        'date': [article['publishedAt'] for article in articles],
        'title': [article['title'] for article in articles],
        'description': [article['description'] for article in articles],
        'content': [article['content'] for article in articles],
        'url': [article['url'] for article in articles]
    }
    df = pd.DataFrame(data)
    return df

# Main function
def main():
    company_name = input("Enter the company name to search: ")
    # print(f"Fetching news articles about '{company_name}' from the past 10 days...")

    # Fetch articles
    articles = fetch_news(company_name, API_KEY)

    if articles:
        print(f"Found {len(articles)} articles!")
        
        # Save articles to a DataFrame
        df = save_to_dataframe(articles)
        
        i=0
        path = 'F:\programming\FK_it\company_news{i}.csv'

        # Save the DataFrame to a CSV file
        if os.path.exists:
            i+=1
        df.to_csv(f'company_news{i}.csv', index=False)
        # print(f"Articles saved to '{company_name}_news.csv'.")
    else:
        print("No articles found.")

if __name__ == "__main__":
    main()
