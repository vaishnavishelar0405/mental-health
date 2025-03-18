import pandas as pd
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

model = joblib.load("mental_health_model.pkl")

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment using VADER and TextBlob."""
    vader_score = analyzer.polarity_scores(str(text))["compound"]
    blob_score = TextBlob(str(text)).sentiment.polarity
    return (vader_score + blob_score) / 2


emotion_text = input("Enter your emotion text: ")
routine_text = input("Enter your routine text: ")
trauma_history = input("Enter trauma history (0 for No, 1 for Yes): ")

# Convert inputs into a DataFrame
input_data = pd.DataFrame({
    "emotion_score": [analyze_sentiment(emotion_text)],
    "routine_score": [analyze_sentiment(routine_text)],
    "trauma_score": [int(trauma_history)]
})

prediction = model.predict(input_data)


print("\n Predicted Mental Health Stage:", prediction[0])
