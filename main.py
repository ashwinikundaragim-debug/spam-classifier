from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model once
model = joblib.load("model.pkl")

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam classifier API"}

@app.post("/predict")
def predict(msg: Message):
    prediction = model.predict([msg.text])[0]
    return {
        "text": msg.text,
        "is_spam": bool(prediction)
    }