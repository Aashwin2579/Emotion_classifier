from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
print("🔄 Downloading dataset...")
dataset = load_dataset("dair-ai/emotion")
train_data = dataset["train"]

# Convert to pandas DataFrame
df = pd.DataFrame(train_data)
df["emotion"] = df["label"].map(lambda x: train_data.features["label"].int2str(x))

# Prepare features and labels
X = df["text"]
y = df["emotion"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_vect, y)

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/tokenizer.pkl")

print("✅ Training complete. Model saved to /model")