import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
df = pd.read_csv("cleaned_data.csv")
print(df.isnull().sum())
df = df.dropna(subset=["clean_comment", "label"])
df["clean_comment"] = df["clean_comment"].astype(str)
X = df["clean_comment"]
y = df["label"]
print("Dataset shape after removing NaN:", df.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Naive Bayes": MultinomialNB(), "Linear SVM": LinearSVC()}
best_model = None
best_accuracy = 0
for name, model in models.items():
    print("=" * 60)
    print(name)
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
print("=" * 60)
print("Best Accuracy:", best_accuracy)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
print("Model Saved Successfully")