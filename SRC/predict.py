import joblib
from pathlib import Path
from preprocess import clean_text
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "vectorizer.pkl")
while True:
    text = input("\nEnter a comment (or type 'exit' to quit): ")
    if text.lower() == "exit":
        print("exited")
        break
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    print("\nPrediction:", prediction)