from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load once when the app starts
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

# Same order as the columns in your training data (do NOT change the order)
# (field name, label shown to user, typical value from your dataset)
FEATURES = [
    ("fixed_acidity",        "Fixed acidity",        7.9),
    ("volatile_acidity",     "Volatile acidity",     0.52),
    ("citric_acid",          "Citric acid",          0.26),
    ("residual_sugar",       "Residual sugar",       2.2),
    ("chlorides",            "Chlorides",            0.079),
    ("free_sulfur_dioxide",  "Free sulfur dioxide",   14),
    ("total_sulfur_dioxide", "Total sulfur dioxide",  38),
    ("density",              "Density",              0.9967),
    ("ph",                   "pH",                   3.31),
    ("sulphates",            "Sulphates",            0.62),
    ("alcohol",              "Alcohol",              10.2),
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    values = {}
    error = None

    if request.method == "POST":
        try:
            row = []
            for name, label, default in FEATURES:
                values[name] = request.form[name]
                row.append(float(request.form[name]))

            data = np.array([row])             # shape (1, 11)
            data_scaled = scaler.transform(data)  # same scaling as training
            result = model.predict(data_scaled)[0]

            if hasattr(model, "predict_proba"):
                confidence = round(max(model.predict_proba(data_scaled)[0]) * 100, 1)
        except ValueError:
            error = "Please enter valid numbers in every field."

    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        confidence=confidence,
        values=values,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
