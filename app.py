import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Sentiment Prediction
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    vectorized = vectorizer.transform([text])
    result = model.predict(vectorized)
    return "Negative" if result == 0 else "Positive"

# Stylish card for displaying sentiment
def create_card(tweet_text, sentiment):
    color = "#4CAF50" if sentiment == "Positive" else "#F44336"
    emoji = "😊" if sentiment == "Positive" else "😞"
    card_html = f"""
    <div style="border: 2px solid {color}; background-color: #f9f9f9; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <h4 style="color: {color}; margin-bottom: 10px;">{emoji} {sentiment} Sentiment</h4>
        <p style="color: #333; font-size: 16px;">{tweet_text}</p>
    </div>
    """
    return card_html

# App UI
def main():
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #3366cc;
            font-size: 40px;
            font-weight: bold;
            margin-top: 10px;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #777;
            margin-bottom: 20px;
        }
        </style>
        <div class='main-title'>🐦 Twitter Sentiment Analyzer</div>
        <div class='subtext'>Analyze your own text or tweets from any public Twitter user</div>
    """, unsafe_allow_html=True)

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.selectbox("Choose an option", ["✏️ Input text", "🐤 Get tweets from user"])

    if option == "✏️ Input text":
        text_input = st.text_area("Enter text to analyze sentiment", height=150, placeholder="Type something like 'I love this product!'")
        if st.button("Analyze Sentiment"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text!")

    elif option == "🐤 Get tweets from user":
        username = st.text_input("Enter Twitter username", placeholder="elonmusk (no @ required)")
        if st.button("Fetch Tweets"):
            if username.strip():
                try:
                    tweets_data = scraper.get_tweets(username, mode='user', number=5)
                    if 'tweets' in tweets_data and tweets_data['tweets']:
                        for tweet in tweets_data['tweets']:
                            tweet_text = tweet['text']
                            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                            st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
                    else:
                        st.warning("No tweets found or the account is private.")
                except Exception as e:
                    st.error(f"Failed to fetch tweets: {str(e)}")
            else:
                st.warning("Please enter a valid username.")

if __name__ == "__main__":
    main()