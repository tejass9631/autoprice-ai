import os
import sys
import warnings
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(
    title="AutoPrice AI Valuation API",
    description="Machine learning valuation engine for used cars using Random Forest, ANN, and KMeans",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Robust dynamic path resolver for local and serverless environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def resolve_model_file(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(ROOT_DIR, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "api", filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename

# Lazy loading container for models
models = {}

def load_models():
    if not models:
        try:
            models["fc"] = joblib.load(resolve_model_file("feature_columns.pkl"))
            models["fc_map"] = {name: i for i, name in enumerate(models["fc"])}
            models["cfc"] = joblib.load(resolve_model_file("cluster_feature_columns.pkl"))
            models["cfc_map"] = {name: i for i, name in enumerate(models["cfc"])}
            models["scaler"] = joblib.load(resolve_model_file("scaler.pkl"))
            models["best_model"] = joblib.load(resolve_model_file("best_model.pkl"))
            models["kmeans"] = joblib.load(resolve_model_file("kmeans_model.pkl"))
            models["scaler_cluster"] = joblib.load(resolve_model_file("cluster_scaler.pkl"))
            models["cluster_names"] = joblib.load(resolve_model_file("cluster_names.pkl"))
            models["ann_model"] = joblib.load(resolve_model_file("ann_model.pkl"))
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            raise e

# Eager load if possible
try:
    load_models()
except Exception as e:
    print(f"Pre-load warning: {e}", file=sys.stderr)


class CarInput(BaseModel):
    brand: str
    model: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str


@app.get("/")
@app.get("/api")
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AutoPrice AI Engine",
        "models_loaded": "best_model" in models
    }


def predict_valuation(car: CarInput):
    load_models()

    fc = models["fc"]
    fc_map = models["fc_map"]
    cfc = models["cfc"]
    cfc_map = models["cfc_map"]
    scaler = models["scaler"]
    best_model = models["best_model"]
    kmeans = models["kmeans"]
    scaler_cluster = models["scaler_cluster"]
    cluster_names = models["cluster_names"]
    ann_model = models["ann_model"]

    # 1. Price Feature Vector (Pure NumPy - avoids single-row dummy drop)
    vec = np.zeros((1, len(fc)), dtype=np.float64)
    if "year" in fc_map:
        vec[0, fc_map["year"]] = float(car.year)
    if "km_driven" in fc_map:
        vec[0, fc_map["km_driven"]] = float(car.km_driven)

    for prefix, val in [
        ("brand", car.brand),
        ("model", car.model),
        ("fuel", car.fuel),
        ("seller_type", car.seller_type),
        ("transmission", car.transmission),
        ("owner", car.owner)
    ]:
        key = f"{prefix}_{val}"
        if key in fc_map:
            vec[0, fc_map[key]] = 1.0

    # Random Forest inference
    price_scaled = scaler.transform(vec)
    rf_pred = float(best_model.predict(price_scaled)[0])

    # 2. ANN Prediction
    W1, b1 = ann_model["W1"], ann_model["b1"]
    W2, b2 = ann_model["W2"], ann_model["b2"]
    y_mean, y_std = ann_model["y_mean"], ann_model["y_std"]

    z1 = np.dot(price_scaled, W1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, W2) + b2
    ann_pred = float(((z2 * y_std) + y_mean).item())

    # 3. Market Clustering
    cvec = np.zeros((1, len(cfc)), dtype=np.float64)
    if "year" in cfc_map:
        cvec[0, cfc_map["year"]] = float(car.year)
    if "km_driven" in cfc_map:
        cvec[0, cfc_map["km_driven"]] = float(car.km_driven)

    for prefix, val in [
        ("fuel", car.fuel),
        ("seller_type", car.seller_type),
        ("transmission", car.transmission),
        ("owner", car.owner)
    ]:
        key = f"{prefix}_{val}"
        if key in cfc_map:
            cvec[0, cfc_map[key]] = 1.0

    cvec_scaled = scaler_cluster.transform(cvec)
    cluster_idx = int(kmeans.predict(cvec_scaled)[0])
    market_segment = str(cluster_names[cluster_idx])

    return {
        "predicted_price": round(rf_pred, 2),
        "ann_prediction": round(ann_pred, 2),
        "market_segment": market_segment,
        "best_model": "Random Forest"
    }


@app.post("/analyze")
@app.post("/api/analyze")
def analyze_endpoint(car: CarInput):
    try:
        return predict_valuation(car)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
