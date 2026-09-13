import os
import sys
import warnings
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(
    title="AutoPrice AI — Full-Stack Valuation Platform",
    description="Machine learning valuation engine with interactive 3D studio and real-time inference",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_file(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "frontend", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "frontend", filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, filename)

# Load Trained ML Models
best_model = joblib.load(resolve_file("best_model.pkl"))
scaler = joblib.load(resolve_file("scaler.pkl"))
feature_columns = joblib.load(resolve_file("feature_columns.pkl"))
fc_map = {name: i for i, name in enumerate(feature_columns)}

kmeans = joblib.load(resolve_file("kmeans_model.pkl"))
scaler_cluster = joblib.load(resolve_file("cluster_scaler.pkl"))
cluster_names = joblib.load(resolve_file("cluster_names.pkl"))
cluster_feature_columns = joblib.load(resolve_file("cluster_feature_columns.pkl"))
cfc_map = {name: i for i, name in enumerate(cluster_feature_columns)}

ann_model = joblib.load(resolve_file("ann_model.pkl"))


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


# ---------------------------------------------------------------------------
# 1. FRONTEND UI & STATIC ASSET ROUTES
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
@app.get("/index.html", response_class=FileResponse)
def serve_home():
    index_file = resolve_file("index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse("<h2>AutoPrice AI Frontend Not Found</h2>", status_code=404)


@app.get("/car_brands_models.js")
def serve_car_brands():
    fpath = resolve_file("car_brands_models.js")
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="car_brands_models.js not found")


@app.get("/car_model_data.js")
def serve_car_model_data():
    fpath = resolve_file("car_model_data.js")
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="car_model_data.js not found")


@app.get("/car.glb")
def serve_car_glb():
    fpath = resolve_file("car.glb")
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="model/gltf-binary")
    raise HTTPException(status_code=404, detail="car.glb not found")


frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


# ---------------------------------------------------------------------------
# 2. HEALTH CHECK & API STATUS
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/api")
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "message": "Used Car Price & Market Analysis API is running"
    }


# ---------------------------------------------------------------------------
# 3. MACHINE LEARNING VALUATION INFERENCE
# ---------------------------------------------------------------------------
@app.post("/analyze")
@app.post("/api/analyze")
def analyze_car(car: CarInput):
    # Security: input boundaries and range validation
    if car.year < 1980 or car.year > 2026:
        raise HTTPException(status_code=422, detail="Manufacture year must be between 1980 and 2026.")
    if car.km_driven < 0 or car.km_driven > 2000000:
        raise HTTPException(status_code=422, detail="Kilometers driven must be between 0 and 2,000,000.")
    if not car.brand or len(car.brand) > 100 or not car.model or len(car.model) > 100:
        raise HTTPException(status_code=422, detail="Invalid brand or model input.")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal calculation error. Please verify vehicle specifications.")
