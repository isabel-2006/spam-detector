from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    prediction = model.predict([message])[0]
    probability = model.predict_proba([message]).max()

    if prediction == "spam":
        result = "Spam 🚫"
    else:
        result = "Not Spam ✅"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability*100,2)
    )

if __name__ == "__main__":
    app.run(debug=True)