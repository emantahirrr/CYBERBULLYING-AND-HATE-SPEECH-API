from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from preprocess import clean_text
model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "vectorizer.pkl")
app = FastAPI(
    title="Cyberbullying & Hate Speech Detection API",
    version="1.0.0"
)
class Comment(BaseModel):
    text: str
@app.get("/")
def home():
    return {
        "message": "API is running"
    }
@app.post("/predict")
def predict(comment: Comment):
    cleaned = clean_text(comment.text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return {
        "input": comment.text,
        "cleaned_text": cleaned,
        "prediction": prediction
    }