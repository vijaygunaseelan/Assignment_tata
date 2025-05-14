from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import pickle
import os
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Load models
models = {
    "logistic": pickle.load(open("/Users/vijayg/Desktop/personal/practice/logistic_regression_model.pkl", "rb")),
    "random_forest": pickle.load(open("/Users/vijayg/Desktop/personal/practice/random_forest_model.pkl", "rb")),
    "mlp": pickle.load(open("/Users/vijayg/Desktop/personal/practice/mlp_classifier_model.pkl", "rb")),
}

# You should define the exact preprocessing steps from training
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID, target, and handle missing values
    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.fillna(method='ffill')

    # Convert categorical variables to dummies
    df = pd.get_dummies(df)
    return df

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    if model not in models:
        return {"error": "Invalid model name. Use 'logistic', 'random_forest', or 'mlp'"}

    df = pd.read_csv(file.file)
    X = preprocess(df)

    # Align features with model (assuming original training features are known)
    model_obj = models[model]
    try:
        predictions = model_obj.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
