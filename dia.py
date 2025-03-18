import nltk
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def analyze_sentiment(text):
   
    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)["compound"]
    blob_score = TextBlob(text).sentiment.polarity
    
    avg_score = (vader_score + blob_score) / 2
    return avg_score

def assess_mental_health():
   
    print("Welcome to the Mental Health AI Assessment")
    
    # 1) behaviour:-
    emotions_input = input("Describe your emotions and feelings recently: ")
    sentiment_score = analyze_sentiment(emotions_input)
    
    # 2Check daily routine:-
    routine_input = input("How do you usually spend your day? ")
    routine_score = analyze_sentiment(routine_input)
    
    # 3)Trauma History
    trauma_input = input("Have you experienced any past trauma or accidents affecting mental health? (Yes/No)")
    trauma_flag = 1 if trauma_input.lower() == "yes" else 0
    
    
    feature_vector = np.array([[sentiment_score, routine_score, trauma_flag]])
    
    # as file is not ready,so handling file:-
    try:
        model = joblib.load("mental_health_model.pkl")
        stage = model.predict(feature_vector)[0]
    except FileNotFoundError:
        print("Warning: Pre-trained model not found. Using basic scoring method.")
        total_score = sentiment_score + routine_score - trauma_flag  # Basic score computation
        
        if total_score > 0.5:
            stage = 1
        elif -0.5 <= total_score <= 0.5:
            stage = 2
        else:
            stage = 3
    
    # provided by me
    stage_descriptions = {
        1: "Early signs of stress or mild anxiety. Suggested self-care and mindfulness activities.",
        2: "Moderate issues detected (overthinking, panic episodes). Suggested structured therapy and engagement.",
        3: "Severe distress detected (depression, PTSD, suicidal thoughts). Professional help is advised."
    }
    
    return {"stage": stage, "status": stage_descriptions[stage]}


diagnosis = assess_mental_health()
print(f"Diagnosis Result: Stage {diagnosis['stage']}")
print(f"Recommendation: {diagnosis['status']}")