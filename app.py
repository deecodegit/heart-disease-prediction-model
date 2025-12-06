from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import json
import os

# create flask app
app = Flask(__name__)

# absolute safe paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "final_pipeline.joblib")
metrics_path = os.path.join(base_dir, "reports", "final_pipeline_metrics.json")

# load model
final_pipeline = joblib.load(model_path)

# load metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # expected order of features
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"
        ]

        # collect data
        values = []
        for name in feature_names:
            val = request.form.get(name)

            if val is None or val.strip() == "":
                raise ValueError(f"missing value for: {name}")

            values.append(float(val))

        # create proper dataframe for pipeline
        input_df = pd.DataFrame([values], columns=feature_names)

        # prediction
        pred = final_pipeline.predict(input_df)[0]

        # probability
        if hasattr(final_pipeline, "predict_proba"):
            prob = float(final_pipeline.predict_proba(input_df)[0][1])
            prob_text = f"{prob * 100:.2f}%"
        else:
            prob_text = "n/a"

        result_text = "high risk ❤️‍🔥" if pred == 1 else "low risk 💙"

        return render_template(
            "result.html",
            result=result_text,
            probability=prob_text,
            values=values
        )

    except Exception as e:
        # show actual error for debugging
        return render_template(
            "result.html",
            result="error",
            probability=str(e),
            values=[]
        )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
