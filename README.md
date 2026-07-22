

The **Cyberbullying & Hate Speech Detection System** is a Natural Language Processing (NLP) and Machine Learning project that classifies online comments into **Hate Speech**, **Offensive**, or **Normal** categories. The system is trained using the **HateXplain dataset**, applies text preprocessing and TF-IDF vectorization, and uses supervised machine learning algorithms to detect harmful online content. The trained model is deployed through a **FastAPI** REST API for real-time predictions.

---

## Features

- Detects Hate Speech, Offensive, and Normal comments.
- Performs text preprocessing and cleaning.
- Uses TF-IDF for feature extraction.
- Trains multiple machine learning models.
- Compares model performance using evaluation metrics.
- Saves the best-performing model for inference.
- Provides real-time predictions through FastAPI.
- Easy to extend with new datasets or deep learning models.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- FastAPI
- Uvicorn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset

**Dataset:** HateXplain Dataset

Files Used:

- `final_hateXplain.csv`
- `hateXplain.csv`

Dataset Columns:

- comment
- label
- Race
- Religion
- Gender
- Sexual Orientation
- Miscellaneous

---

## Project Structure

```
CyberbullyingDetection/
│
├── app/
│   └── main.py
│
├── data/
│   ├── final_hateXplain.csv
│   ├── hateXplain.csv
│   └── cleaned_data.csv
│
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. Load the HateXplain dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Clean and preprocess text.
4. Convert text into TF-IDF vectors.
5. Split data into training and testing sets.
6. Train multiple machine learning models.
7. Evaluate model performance.
8. Save the best model and vectorizer.
9. Deploy the model using FastAPI.
10. Predict labels for new user comments.

---

## Text Preprocessing

The preprocessing pipeline includes:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Remove numbers
- Remove extra spaces
- Remove stopwords
- Lemmatization

---

## Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)

The best-performing model is automatically saved for prediction.

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/CyberbullyingDetection.git
```

Navigate to the project directory:

```bash
cd CyberbullyingDetection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python src/train.py
```

The script will:

- Load the cleaned dataset
- Train multiple classifiers
- Evaluate performance
- Save:
  - `best_model.pkl`
  - `vectorizer.pkl`

---

## Running Predictions

Run:

```bash
python src/predict.py
```

Example:

```
Enter a comment:

You are an idiot
```

Output:

```
Prediction:
Offensive
```

---

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Use the Swagger interface to test predictions.

---

## API Endpoints

### GET /

Returns API status.

Response

```json
{
  "message": "API is running"
}
```

---

### POST /predict

Request

```json
{
  "text": "You are stupid."
}
```

Response

```json
{
  "input": "You are stupid.",
  "cleaned_text": "stupid",
  "prediction": "Offensive"
}
```

---

## Future Improvements

- Deep Learning (LSTM, GRU)
- BERT/DistilBERT Integration
- Explainable AI
- Streamlit Web Interface
- Docker Support
- Cloud Deployment
- Multi-language Support
- User Authentication
- Database Integration

---

## Applications

- Social Media Moderation
- Online Community Management
- Content Filtering
- Educational Platforms
- Discussion Forums
- Gaming Platforms
- Comment Moderation Systems

---


## License

This project is developed for educational and research purposes.
