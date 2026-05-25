print("Script started")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

print("Loading dataset...")

data = pd.read_csv("dataset/SMSSpamCollection", sep="\t", names=["label","message"])

print("Dataset loaded!")
print(data.head())

X = data["message"]
y = data["label"]

print("Training model...")

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

print("Saving model...")

pickle.dump(model, open("spam_model.pkl","wb"))

print("Model trained successfully!")
