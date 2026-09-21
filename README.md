# Student Placement Prediction System

A machine-learning web application that predicts whether a student is likely to be placed based on academic performance, skills, internships, projects, attendance, and certifications.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Flask
- HTML/CSS

## Features
- Placement classification: Placed / Not Placed
- Placement probability
- Simple web interface
- Reproducible model-training script

## Project Structure
```text
student-placement-prediction/
├── data/placement_data.csv
├── model/placement_model.pkl
├── src/train_model.py
├── templates/index.html
├── static/style.css
├── static/script.js
├── app.py
├── requirements.txt
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
python src/train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Dataset Note
The included dataset is **synthetic**, created for academic/demo purposes. It should not be presented as real institutional placement data.
