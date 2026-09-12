import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AutoPrice AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load Models
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
fc_map = {name: i for i, name in enumerate(feature_columns)}

kmeans = joblib.load("kmeans_model.pkl")
scaler_cluster = joblib.load("cluster_scaler.pkl")
cluster_names = joblib.load("cluster_names.pkl")
cluster_feature_columns = joblib.load("cluster_feature_columns.pkl")
cfc_map = {name: i for i, name in enumerate(cluster_feature_columns)}

ann_model = joblib.load("ann_model.pkl")


class CarInput(BaseModel):
    brand: str
    model: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@app.get("/")
@app.get("/health")
def home():
    return {
        "status": "healthy",
        "message": "Used Car Price & Market Analysis API is running"
    }


@app.post("/analyze")
@app.post("/api/analyze")
def analyze_car(car: CarInput):
    try:
        # Price prediction feature vector
        vec = np.zeros((1, len(feature_columns)), dtype=np.float64)
        if "year" in fc_map:
            vec[0, fc_map["year"]] = float(car.year)
        if "km_driven" in fc_map:
            vec[0, fc_map["km_driven"]] = float(car.km_driven)

        for col_prefix, val in [
            ("brand", car.brand),
            ("model", car.model),
            ("fuel", car.fuel),
            ("seller_type", car.seller_type),
            ("transmission", car.transmission),
            ("owner", car.owner)
        ]:
            key = f"{col_prefix}_{val}"
            if key in fc_map:
                vec[0, fc_map[key]] = 1.0

        price_scaled = scaler.transform(vec)
        rf_prediction = best_model.predict(price_scaled)[0]

        # ANN prediction
        W1 = ann_model["W1"]
        b1 = ann_model["b1"]
        W2 = ann_model["W2"]
        b2 = ann_model["b2"]
        y_mean = ann_model["y_mean"]
        y_std = ann_model["y_std"]

        z1 = np.dot(price_scaled, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2

        ann_prediction = (z2 * y_std) + y_mean

        # K-Means market segment
        cvec = np.zeros((1, len(cluster_feature_columns)), dtype=np.float64)
        if "year" in cfc_map:
            cvec[0, cfc_map["year"]] = float(car.year)
        if "km_driven" in cfc_map:
            cvec[0, cfc_map["km_driven"]] = float(car.km_driven)

        for col_prefix, val in [
            ("fuel", car.fuel),
            ("seller_type", car.seller_type),
            ("transmission", car.transmission),
            ("owner", car.owner)
        ]:
            key = f"{col_prefix}_{val}"
            if key in cfc_map:
                cvec[0, cfc_map[key]] = 1.0

        cluster_scaled = scaler_cluster.transform(cvec)
        cluster = kmeans.predict(cluster_scaled)[0]
        market_segment = cluster_names[cluster]

        return {
            "predicted_price": round(float(rf_prediction), 2),
            "ann_prediction": round(float(ann_prediction[0][0]), 2),
            "market_segment": market_segment,
            "best_model": "Random Forest"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
