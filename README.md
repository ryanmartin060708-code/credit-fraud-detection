# 🛡️ FraudSense AI — Credit Card Fraud Detection Dashboard

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-supported-blue)

An **end-to-end, production-ready machine learning application** that detects fraudulent credit card transactions using an ensemble of supervised learning models, interactive Plotly visualisations, SHAP explainability, and a professional Streamlit dashboard.

---

## 📸 Screenshots

| Home Dashboard | EDA | Model Performance | SHAP |
|---|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## ✨ Features

| Category | Details |
|---|---|
| **Data** | Kaggle Credit Card Fraud dataset · duplicate removal · missing-value handling |
| **Imbalance** | SMOTE · Random Undersampling · Class Weighting (toggleable) |
| **Models** | Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · XGBoost · LightGBM |
| **Tuning** | Optuna hyperparameter optimisation (30–100 trials) |
| **Metrics** | Accuracy · Precision · Recall · F1 · ROC AUC · PR AUC |
| **Charts** | ROC curve · PR curve · Confusion matrix · Feature importance |
| **Explainability** | SHAP summary · waterfall · force plots |
| **Prediction** | Manual single-transaction entry with risk gauge |
| **Batch** | Upload CSV → download annotated predictions |
| **DevOps** | Docker · CLI entrypoint · modular `src/` package |

---

## 🗂️ Project Structure

```
credit-fraud-detection/
│
├── app/
│   ├── Home.py                  # Landing page & KPIs
│   └── pages/
│       ├── 1_EDA.py             # Exploratory Data Analysis
│       ├── 2_Model_Training.py  # Train, compare & tune models
│       ├── 3_Predict.py         # Single & batch prediction
│       ├── 4_Model_Performance.py # Deep-dive evaluation
│       └── 5_SHAP_Explainability.py # SHAP plots
│
├── data/
│   └── creditcard.csv           # ← Place dataset here
│
├── models/
│   ├── best_model.pkl           # Auto-saved after training
│   └── scaler.pkl
│
├── src/
│   ├── preprocess.py            # Cleaning, scaling, splitting
│   ├── train.py                 # Model training + Optuna
│   ├── evaluate.py              # Metrics & Plotly charts
│   ├── predict.py               # Inference utilities
│   ├── explain.py               # SHAP wrappers
│   └── utils.py                 # Shared constants & helpers
│
├── main.py                      # CLI pipeline runner
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ryan.martin060708-code/credit-fraud-detection.git
cd credit-fraud-detection
python -m venv venv
pip install -r requirements.txt
```

### 2. Download the dataset

Download `creditcard.csv` from  
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Place it at:
```
data/creditcard.csv
```

### 3. (Optional) Run the CLI pipeline

Train all models from the terminal:

```bash
python main.py
# or with custom settings:
python main.py --imbalance SMOTE --test-size 0.2
```

### 4. Launch the dashboard

```bash
streamlit run app/Home.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker

```bash
# Build
docker build -t fraud-dashboard .

# Run (mount your dataset)
docker run -p 8501:8501 -v $(pwd)/data:/app/data fraud-dashboard
```

---

## 🤖 Model Pipeline

```
Raw CSV
  └─ load_and_clean()         remove duplicates, handle NaN
       └─ split_data()        stratified 80/20 train-test split
            └─ scale_features()    StandardScaler (fit on train only)
                 └─ apply_imbalance_strategy()  SMOTE / under-sampling
                      └─ train_all()            6 classifiers
                           └─ evaluate_all()    full metrics suite
                                └─ save_best_model()  best by ROC AUC
```

---

## 📊 Performance Metrics (example — varies by run)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| LightGBM | 0.9996 | 0.9524 | 0.8367 | 0.8908 | **0.9988** |
| XGBoost | 0.9995 | 0.9231 | 0.8571 | 0.8889 | 0.9985 |
| Random Forest | 0.9994 | 0.9600 | 0.7959 | 0.8702 | 0.9974 |
| Gradient Boosting | 0.9993 | 0.9355 | 0.7959 | 0.8600 | 0.9966 |
| Logistic Regression | 0.9990 | 0.8788 | 0.7449 | 0.8062 | 0.9892 |
| Decision Tree | 0.9989 | 0.8235 | 0.7143 | 0.7650 | 0.8553 |

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) provides game-theoretic guarantees of feature attribution:

- **TreeExplainer** is used for all tree-based models (fast, exact)
- **KernelExplainer** is used for Logistic Regression (approximate)
- V4, V14, V12, and V10 consistently rank as the most influential features

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| ML | scikit-learn · XGBoost · LightGBM · imbalanced-learn |
| Tuning | Optuna |
| Explainability | SHAP |
| Visualisation | Plotly · Matplotlib · Seaborn |
| Dashboard | Streamlit |
| Persistence | Joblib |
| Containerisation | Docker |

---

## 🔮 Future Improvements

- [ ] Real-time Kafka/streaming transaction scoring
- [ ] FastAPI REST endpoint for model serving
- [ ] Drift detection with Evidently AI
- [ ] AutoML integration (FLAML / AutoGluon)
- [ ] PostgreSQL model registry
- [ ] CI/CD with GitHub Actions
- [ ] MLflow experiment tracking

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

*Built with ❤️ using Streamlit, Scikit-learn, XGBoost, LightGBM, SHAP, and Optuna.*
