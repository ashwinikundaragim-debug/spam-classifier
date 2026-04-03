from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset
texts = [
    "Win money now",
    "Limited offer just for you",
    "Hey, how are you?",
    "Let's meet tomorrow",
]
labels = [1, 1, 0, 0]  # 1 = spam, 0 = not spam

# Pipeline = vectorizer + model
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("model", LogisticRegression())
])

pipeline.fit(texts, labels)

joblib.dump(pipeline, "model.pkl")
print("Model saved!")