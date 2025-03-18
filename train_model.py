import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# df = pd.read_csv("mental_health_data_final.csv")
df = pd.read_csv(r"C:\Users\omkur\OneDrive\Desktop\Project\mental_health_data_final.csv")


analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(row):
    """Analyze sentiment using VADER and TextBlob for multiple columns."""
    vader_score = np.mean([analyzer.polarity_scores(str(text))['compound'] for text in row if pd.notna(text)])
    blob_score = np.mean([TextBlob(str(text)).sentiment.polarity for text in row if pd.notna(text)])
    return (vader_score + blob_score) / 2


df['emotion_score'] = df[['Emotion_1', 'Emotion_2', 'Emotion_3', 'Emotion_4', 'Emotion_5', 'Emotion_6', 'Emotion_7']].apply(analyze_sentiment, axis=1)
df['routine_score'] = df[['Routine_1', 'Routine_2', 'Routine_3', 'Routine_4', 'Routine_5', 'Routine_6', 'Routine_7']].apply(analyze_sentiment, axis=1)

df['trauma_score'] = df[['Trauma_1', 'Trauma_2', 'Trauma_3', 'Trauma_4', 'Trauma_5', 'Trauma_6', 'Trauma_7']].mean(axis=1)  # Assuming trauma scores are numeric

X = df[['emotion_score', 'routine_score', 'trauma_score']]
y = np.random.choice([0, 1, 2], size=len(df)) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "mental_health_model.pkl")

print("Model trained and saved successfully!")



