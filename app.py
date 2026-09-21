from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(BASE / "model" / "placement_model.pkl")

FEATURES = [
    "cgpa","tenth_percentage","twelfth_percentage","backlogs",
    "internships","projects","coding_score","aptitude_score",
    "communication_score","attendance","certifications"
]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    if request.method == "POST":
        values = [float(request.form[f]) for f in FEATURES]
        X = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(X)[0])
        probability = round(float(model.predict_proba(X)[0][1]) * 100, 1)
        result = "Placed" if prediction else "Not Placed"
    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
