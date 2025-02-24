import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime

# Text processing & sentiment analysis
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Machine learning tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For combining additional features with sparse TF-IDF matrix
from scipy.sparse import hstack, csr_matrix

# WordCloud for visualization
from wordcloud import WordCloud

# For real-time stock price
import yfinance as yf

# Download necessary NLTK data (only once)
nltk.download('stopwords')
nltk.download('vader_lexicon')

# =============================================================================
# Step 1: Fetch News Online from NewsAPI
# =============================================================================
def fetch_news(query, page_size=50):
    """
    Fetch news articles online using NewsAPI.
    Replace 'YOUR_NEWSAPI_KEY' with your actual API key from https://newsapi.org/.
    """
    api_key = "64d7421419c64411a5d2d5f4aa39d05e"  # <-- Replace with your NewsAPI key
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"pageSize={page_size}&"
        "sortBy=publishedAt&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        news_list = []
        for article in articles:
            published_at = article.get("publishedAt", None)
            title = article.get("title", "")
            description = article.get("description", "")
            # Combine title and description as news content
            news_content = f"{title} {description}"
            news_list.append({"date": published_at, "news": news_content})
        return news_list
    else:
        print("Error fetching news:", response.status_code)
        return []

# =============================================================================
# Step 2: Prepare the Data
# =============================================================================
# Define the companies you are interested in:
companies = ['Apple', 'Google', 'Amazon', 'Tesla', 'Microsoft']
all_news = []

# Fetch news for each company and tag the data accordingly
for company in companies:
    print(f"Fetching news for {company}...")
    news_items = fetch_news(query=company, page_size=50)
    for item in news_items:
        item['company'] = company
        all_news.append(item)

# Create a DataFrame from the fetched news
news_data = pd.DataFrame(all_news)

# Check if any news was fetched
if news_data.empty:
    print("No news data fetched. Check your API key or internet connection.")
    # Optionally, exit the script
    import sys
    sys.exit("Exiting due to no fetched news data.")
else:
    # Convert 'date' to datetime if data is available
    news_data['date'] = pd.to_datetime(news_data['date'])

# Output the fetched data
print("\n--- Fetched News Data (first 5 rows) ---")
print(news_data.head())

# =============================================================================
# Step 3: Preprocess and Compute Sentiment
# =============================================================================
def preprocess_text(text):
    """
    Lowercase, remove non-alphabetical characters,
    and remove English stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

def compute_sentiment(text, analyzer):
    """
    Compute the compound sentiment score using VADER and scale it
    from [-1, 1] to [-5, 5].
    """
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    scaled_score = compound_score * 5
    return scaled_score

# Clean the news text
news_data['clean_news'] = news_data['news'].apply(preprocess_text)

# Initialize VADER sentiment analyzer and compute sentiment scores
analyzer = SentimentIntensityAnalyzer()
news_data['sentiment_score'] = news_data['news'].apply(lambda x: compute_sentiment(x, analyzer))

# Add a polarity column based on sentiment score
def get_polarity(score):
    if score >= 1.0:
        return "Positive"
    elif score <= -1.0:
        return "Negative"
    else:
        return "Neutral"

news_data['polarity'] = news_data['sentiment_score'].apply(get_polarity)

# =============================================================================
# Step 4: Simulate Impact and Duration (for demonstration purposes)
# =============================================================================
# (In a real application, you would have historical labels.)
np.random.seed(42)
noise = np.random.normal(0, 1, size=len(news_data))
news_data['impact'] = news_data['sentiment_score'] + noise
news_data['impact'] = news_data['impact'].clip(-5, 5)  # Ensure within -5 to 5

news_data['duration'] = (np.abs(news_data['sentiment_score']) * 2 +
                         np.random.normal(0, 1, size=len(news_data))).round().astype(int)
news_data['duration'] = news_data['duration'].apply(lambda x: max(1, x))  # at least 1 day

# =============================================================================
# Step 4.1: Add Additional Text Features
# =============================================================================
# These additional features may improve forecast accuracy.
news_data['news_length'] = news_data['clean_news'].apply(lambda x: len(x))
news_data['word_count'] = news_data['clean_news'].apply(lambda x: len(x.split()))
news_data['exclamation_count'] = news_data['news'].apply(lambda x: x.count('!'))

# =============================================================================
# Step 5: Model Development with Additional Features
# =============================================================================
# Feature extraction using TF-IDF on the cleaned news text
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(news_data['clean_news'])

# Create a matrix of additional features and convert to sparse format.
additional_features = news_data[['news_length', 'word_count', 'exclamation_count']].values
additional_features_sparse = csr_matrix(additional_features)

# Combine TF-IDF features and additional features
X_combined = hstack([X_text, additional_features_sparse])

y_impact = news_data['impact']
y_duration = news_data['duration']

# Split into training and test sets
X_train, X_test, y_impact_train, y_impact_test, y_duration_train, y_duration_test = train_test_split(
    X_combined, y_impact, y_duration, test_size=0.2, random_state=42
)

# Train a regression model to predict the "impact"
impact_model = LinearRegression()
impact_model.fit(X_train, y_impact_train)
y_impact_pred = impact_model.predict(X_test)

# Train a regression model to predict the "duration"
duration_model = LinearRegression()
duration_model.fit(X_train, y_duration_train)
y_duration_pred = duration_model.predict(X_test)

# Evaluate the models
impact_mse = mean_squared_error(y_impact_test, y_impact_pred)
impact_r2 = r2_score(y_impact_test, y_impact_pred)
duration_mse = mean_squared_error(y_duration_test, y_duration_pred)
duration_r2 = r2_score(y_duration_test, y_duration_pred)

print("\n--- Model Evaluation ---")
print(f"Impact Model - MSE: {impact_mse:.2f}, R2: {impact_r2:.2f}")
print(f"Duration Model - MSE: {duration_mse:.2f}, R2: {duration_r2:.2f}")

# =============================================================================
# Step 6: Forecast for Today's / Latest News with Impact Duration
# =============================================================================
# Use today's date (normalized to midnight)
today = pd.Timestamp('today').normalize()
# Filter for news published today. If none, take the latest 5 articles.
forecast_data = news_data[news_data['date'].dt.normalize() == today]
if forecast_data.empty:
    forecast_data = news_data.sort_values(by='date', ascending=False).head(5)

# Build forecast feature matrix:
# 1. Transform clean news with TF-IDF
X_forecast_tfidf = vectorizer.transform(forecast_data['clean_news'])
# 2. Build additional features matrix for the forecast data
forecast_additional_features = forecast_data[['news_length', 'word_count', 'exclamation_count']].values
forecast_additional_features_sparse = csr_matrix(forecast_additional_features)
# 3. Combine both
X_forecast = hstack([X_forecast_tfidf, forecast_additional_features_sparse])

# Predict impact and duration for the forecast data
forecast_data = forecast_data.copy()  # To avoid SettingWithCopyWarning
forecast_data['predicted_impact'] = impact_model.predict(X_forecast)
forecast_data['predicted_duration'] = duration_model.predict(X_forecast)

# Convert predicted duration to integer days (rounding)
forecast_data['predicted_duration_int'] = forecast_data['predicted_duration'].round().astype(int)

# Calculate impact start (assume news publication date) and impact end (start date + duration)
forecast_data['impact_start'] = forecast_data['date']
forecast_data['impact_end'] = forecast_data['impact_start'] + pd.to_timedelta(forecast_data['predicted_duration_int'], unit='d')

# Create a new column that combines date and polarity for quick reference
forecast_data['polarity_date'] = forecast_data['date'].dt.strftime('%Y-%m-%d') + " (" + forecast_data['polarity'] + ")"

print("\n--- Forecast for Today's/Latest News ---")
print(forecast_data[['date', 'company', 'news', 'sentiment_score', 'polarity', 'polarity_date', 
                       'predicted_impact', 'predicted_duration_int', 'impact_start', 'impact_end']].to_string(index=False))

# =============================================================================
# Step 6.5: Forecast Impact Timeline Visualization
# =============================================================================
plt.figure(figsize=(10, 6))
for i, row in forecast_data.iterrows():
    # Plot a horizontal line representing the impact period.
    plt.hlines(y=i, xmin=row['impact_start'], xmax=row['impact_end'], color='blue', linewidth=4)
    # Mark the start and end points.
    plt.plot(row['impact_start'], i, "go", label="Impact Start" if i == forecast_data.index[0] else "")
    plt.plot(row['impact_end'], i, "ro", label="Impact End" if i == forecast_data.index[0] else "")
    # Label the bar with the company name.
    plt.text(row['impact_start'], i, f" {row['company']}", va='center', fontsize=10)
plt.xlabel("Date")
plt.title("Forecast Impact Timeline for Latest News")
plt.legend()
plt.yticks([])
plt.tight_layout()
plt.show()

# =============================================================================
# Step 6.6: Real-Time Stock Price for Apple (Example)
# =============================================================================
apple_ticker = yf.Ticker("AAPL")
apple_price = apple_ticker.history(period="1d")['Close'][0]
print("\n--- Real-Time Stock Price ---")
print(f"Current Apple Stock Price: ${apple_price:.2f}")

# =============================================================================
# Step 7: Analysis and Visualization
# =============================================================================
# Visualization 1: Time Series Plot of Simulated Stock Price (for Apple)
apple_dates = pd.date_range(start="2024-03-01", periods=200, freq='D')
np.random.seed(42)
price_changes = np.random.normal(0, 1, size=len(apple_dates))
apple_prices = 150 + np.cumsum(price_changes)

plt.figure(figsize=(10, 6))
plt.plot(apple_dates, apple_prices, marker='o', linestyle='-')
plt.title("Simulated Apple Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Distribution of News Sentiment Scores
plt.figure(figsize=(8, 5))
sns.histplot(news_data['sentiment_score'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of News Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()

# Visualization 3: Scatter Plot of Sentiment Score vs. Impact
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sentiment_score', y='impact', data=news_data, hue='company')
plt.title("Sentiment Score vs. Stock Impact")
plt.xlabel("Sentiment Score")
plt.ylabel("Impact (Scaled -5 to 5)")
plt.legend(title="Company")
plt.show()

# Visualization 4: Scatter Plot of Sentiment Score vs. Duration
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sentiment_score', y='duration', data=news_data, hue='company')
plt.title("Sentiment Score vs. Duration of Impact")
plt.xlabel("Sentiment Score")
plt.ylabel("Duration (days)")
plt.legend(title="Company")
plt.show()

# Visualization 5: Box Plot of Sentiment Score by Company
plt.figure(figsize=(10, 6))
sns.boxplot(x='company', y='sentiment_score', data=news_data)
plt.title("News Sentiment Scores by Company")
plt.xlabel("Company")
plt.ylabel("Sentiment Score")
plt.show()

# Visualization 6: Bar Plot of Average Impact by Company
avg_impact = news_data.groupby('company')['impact'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='company', y='impact', data=avg_impact, palette="viridis")
plt.title("Average Stock Impact by Company")
plt.xlabel("Company")
plt.ylabel("Average Impact")
plt.show()

# Visualization 7: Correlation Heatmap
corr = news_data[['sentiment_score', 'impact', 'duration']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Visualization 8: Distribution of Impact Duration
plt.figure(figsize=(8, 5))
sns.histplot(news_data['duration'], bins=range(1, news_data['duration'].max()+2), color='salmon', kde=False)
plt.title("Distribution of Impact Duration")
plt.xlabel("Duration (days)")
plt.ylabel("Count")
plt.show()

# Visualization 9: Word Cloud for Positive News
positive_text = " ".join(news_data[news_data['sentiment_score'] > 2]['clean_news'])
if positive_text.strip():
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.title("Word Cloud for Positive News")
    plt.axis('off')
    plt.show()
else:
    print("No positive news found for word cloud visualization.")

# Visualization 10: Word Cloud for Negative News
negative_text = " ".join(news_data[news_data['sentiment_score'] < -2]['clean_news'])
if negative_text.strip():
    wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.title("Word Cloud for Negative News")
    plt.axis('off')
    plt.show()
else:
    print("No negative news found for word cloud visualization.")

# Visualization 11: Actual vs. Predicted Impact
plt.figure(figsize=(8, 5))
plt.scatter(y_impact_test, y_impact_pred, color='purple', alpha=0.7)
plt.title("Actual Impact vs. Predicted Impact")
plt.xlabel("Actual Impact")
plt.ylabel("Predicted Impact")
plt.plot([-5, 5], [-5, 5], color='red', linestyle='--')  # Ideal line
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

# Visualization 12: Actual vs. Predicted Duration
plt.figure(figsize=(8, 5))
plt.scatter(y_duration_test, y_duration_pred, color='green', alpha=0.7)
plt.title("Actual Duration vs. Predicted Duration")
plt.xlabel("Actual Duration (days)")
plt.ylabel("Predicted Duration (days)")
min_val = min(min(y_duration_test), min(y_duration_pred))
max_val = max(max(y_duration_test), max(y_duration_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.show()
