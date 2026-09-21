import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE / "data" / "placement_data.csv")

features = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "internships", "projects", "coding_score", "aptitude_score",
    "communication_score", "attendance", "certifications"
]
X = df[features]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred))

joblib.dump(model, BASE / "model" / "placement_model.pkl")
print("Model saved to model/placement_model.pkl")
