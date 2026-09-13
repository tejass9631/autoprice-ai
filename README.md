# 🏎️ AutoPrice AI — Used Car Valuation & Market Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![GitHub Pages](https://img.shields.io/badge/Deploy-GitHub_Pages-222222?style=for-the-badge&logo=githubpages&logoColor=white)](https://pages.github.com/)

> **AutoPrice AI** is a modern, full-stack used car valuation and market intelligence application. It combines an interactive **Three.js 3D car studio**, **dual machine learning models (Random Forest + Artificial Neural Network)**, and **K-Means market clustering** to deliver instant, high-precision vehicle pricing in Indian Rupees (₹).

---

## ✨ Features

- **🎮 Interactive 3D Vehicle Studio**:
  - Real-time 3D Porsche GLB rendering powered by Three.js and DRACOLoader.
  - Interactive camera orbit controls, zoom, pan, and real-time lighting with metallic car paint shaders.
  - Live color palette selector (Jet Black, Guard Red, Gentian Blue, Carrara White, Agate Grey, Python Green).
  - Studio ground shadow, reflection plane, and camera reset controls.

- **🤖 Dual Machine Learning Valuation Engine**:
  - **Random Forest Regressor**: Primary high-precision estimator trained on 4,300+ real market sales records across 1,500+ one-hot encoded vehicle specifications.
  - **Artificial Neural Network (ANN)**: Secondary verification model computing non-linear hidden layer activations.
  - **K-Means Clustering**: Classifies vehicles into distinct market segments: **Budget**, **Mid-Range**, or **Premium**.

- **⚡ Modern Responsive UI**:
  - Porsche luxury-inspired aesthetic (warm cognac accents, frosted glassmorphism, animated odometer counters).
  - Dynamic brand and model dropdown filters matching 25+ major automotive manufacturers.
  - Interactive market tier gauge bar and vehicle summary badge.
  - Zero-latency client fallback engine for instant offline valuations.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript (ES6+) |
| **3D Engine** | Three.js (r128), GLTFLoader, DRACOLoader, OrbitControls |
| **Backend API** | FastAPI, Pydantic, Uvicorn (ASGI) |
| **Machine Learning** | Scikit-Learn (Random Forest, KMeans, StandardScaler), NumPy, Joblib |
| **Cloud & Deployment**| GitHub Pages, GitHub Actions |

---

## 📁 Project Structure

```text
autoprice-ai/
├── .github/
│   └── workflows/pages.yml          # GitHub Pages deployment workflow
├── frontend/
│   ├── index.html                   # Interactive studio & valuation UI
│   ├── car.glb                      # 3D vehicle model
│   ├── car_brands_models.js         # Dataset brands & models list
│   └── car_model_data.js            # Fallback embedded model data
├── app.py                           # Standalone local FastAPI server
├── index.html                       # Web application entry point (GitHub Pages)
├── best_model.pkl                   # Trained Random Forest Regressor
├── scaler.pkl                       # StandardScaler for features
├── feature_columns.pkl              # Feature column mapping (1531 columns)
├── kmeans_model.pkl                 # K-Means clustering model
├── cluster_scaler.pkl               # StandardScaler for clustering
├── cluster_names.pkl                # Segment label mappings
├── cluster_feature_columns.pkl      # Cluster feature schema
├── ann_model.pkl                    # Custom ANN weights & biases
├── requirements.txt                 # Local development Python dependencies
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Quick Start (Run Locally)

### 1. Clone Repository
```bash
git clone https://github.com/tejass9631/autoprice-ai.git
cd autoprice-ai
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend
```bash
uvicorn app:app --reload --port 8000
```
API Documentation will be live at: `http://127.0.0.1:8000/docs`

### 4. Open the Frontend
Open `index.html` (or `frontend/index.html`) in your browser, or run a lightweight static server:
```bash
npx serve .
# or
python -m http.server 3000
```
Visit `http://localhost:3000` to interact with the 3D car studio and valuation engine.

---

## 🌐 API Specification

### `POST /analyze` or `/api/analyze`

**Request Body**:
```json
{
  "brand": "Maruti",
  "model": "Swift",
  "year": 2018,
  "km_driven": 45000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner"
}
```

**Response**:
```json
{
  "predicted_price": 432394.58,
  "ann_prediction": 463865.91,
  "market_segment": "Budget",
  "best_model": "Random Forest"
}
```

---

## 🚀 Deploy to GitHub Pages

1. Push your repository to GitHub.
2. Go to **Settings** > **Pages** in your repository.
3. Under **Build and deployment** > **Source**, select **GitHub Actions**.
4. The workflow in `.github/workflows/pages.yml` will automatically build and deploy your application live!

---

## 📄 License

Distributed under the MIT License. Feel free to use and modify for personal or commercial projects.
