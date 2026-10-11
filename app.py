from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Perceptron model and scaler
model = joblib.load("Thyroid.pkl")
scaler = joblib.load("scaler.pkl")

# Exact column order used during training (read from the scaler, so it always matches)
FEATURES = list(scaler.feature_names_in_)

# Lab tests the page may leave blank, and the "measured" flag that goes with each one.
# T3 has only a flag in this model (no value column).
LABS = {"TSH": "TSH measured", "TT4": "TT4 measured", "T4U": "T4U measured", "FTI": "FTI measured"}

# Average value of each column in the training data. When a lab test wasn't done,
# we use this average instead of 0, because the model never saw 0 for a missing test.
TRAIN_MEAN = dict(zip(FEATURES, scaler.mean_))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    row = {name: float(request.form[name]) for name in FEATURES}

    # A lab that wasn't measured gets the training average, not 0
    for value_col, flag_col in LABS.items():
        if row[flag_col] == 0:
            row[value_col] = TRAIN_MEAN[value_col]

    features = pd.DataFrame([row], columns=FEATURES)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 0:
        result = "Class 0 (Negative)"
    else:
        result = "Class 1 (Positive)"

    # The frontend reads this response and displays the result screen.
    return f"<h1>Prediction Result: {result}</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)