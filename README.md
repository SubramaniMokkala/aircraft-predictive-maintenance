# Aircraft Predictive Maintenance System

ML-powered system for predicting aircraft turbofan engine failures using time-series sensor data and health monitoring.

## 🎯 Project Overview

This project implements machine learning models to analyze aircraft engine sensor data and predict Remaining Useful Life (RUL), enabling proactive maintenance decisions.

**Key Achievement:** LSTM deep learning model achieves **25.00 RMSE** (31.5% better than classical ML)

## 📊 Dataset

- **Source:** NASA Turbofan Engine Degradation Simulation
- **Engines:** 100 run-to-failure samples
- **Sensors:** 21 measurements (temperature, pressure, vibration, etc.)
- **Target:** Remaining Useful Life in flight cycles

## 🚀 Models Implemented

### 1. Classical Machine Learning
- **Linear Regression** (Baseline): 43.38 RMSE
- **Random Forest**: 36.51 RMSE
- **Gradient Boosting**: 37.60 RMSE

### 2. Deep Learning (BETTER MODEL)
- **LSTM Neural Network**: **25.00 RMSE** ⭐
  - 2-layer LSTM architecture
  - 30-cycle sequence learning
  - 33,953 trainable parameters
  - 31.5% improvement over Random Forest

## 📈 Key Features

- **Feature Engineering:** Rolling window averages capture degradation trends
- **Feature Importance:** Rolling features contribute 81% of predictive power
- **Interactive Dashboard:** Streamlit web app for live predictions
- **Model Comparison:** Comprehensive evaluation across metrics

## 🛠️ Tech Stack

- Python 3.13
- TensorFlow/Keras (Deep Learning)
- Scikit-learn (Classical ML)
- Pandas, NumPy (Data Processing)
- Streamlit (Dashboard)
- Plotly (Visualizations)

## 📦 Setup Instructions
```bash
# Clone repository
git clone https://github.com/SubramaniMokkala/aircraft-predictive-maintenance.git
cd aircraft-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NASA dataset
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/ folder

# Run dashboard
streamlit run dashboards/app.py
```

## 📁 Project Structure
```
aircraft-predictive-maintenance/
├── data/                          # NASA dataset
├── notebooks/                     # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_lstm_deep_learning.ipynb
├── models/                        # Trained models
│   ├── random_forest_model.pkl
│   ├── lstm_model.keras
│   └── scaler.pkl
├── dashboards/                    # Streamlit app
│   └── app.py
└── requirements.txt
```

## 🎨 Dashboard Features

- **Live Predictions:** Select engines and get instant RUL forecasts
- **Model Comparison:** Visual performance metrics
- **Interactive Demo:** Test both LSTM and Random Forest models
- **Health Alerts:** Color-coded maintenance recommendations

## 📊 Results Summary

| Model | RMSE (cycles) | MAE (cycles) | R² Score |
|-------|---------------|--------------|----------|
| **LSTM** | **25.00** | **16.92** | **0.8366** |
| Random Forest | 36.51 | 25.51 | 0.7082 |
| Gradient Boosting | 37.60 | 26.68 | 0.6906 |
| Linear Regression | 43.38 | 33.18 | 0.5881 |

## 💼 Business Impact

- **Prediction Accuracy:** ~25 cycles (2-3 weeks advance warning)
- **Proactive Maintenance:** Schedule repairs before failure
- **Reduced Downtime:** Prevent unexpected breakdowns
- **Cost Savings:** Optimize maintenance operations

## 🎓 Skills Demonstrated

- Time-series analysis and forecasting
- Deep learning (LSTM) for sequential data
- Classical ML (ensemble methods)
- Feature engineering and selection
- Model evaluation and comparison
- Interactive dashboard development
- Production ML deployment

## 📧 Contact

**Subramani Mokkala**  
GitHub: [SubramaniMokkala](https://github.com/SubramaniMokkala)  
Project: [aircraft-predictive-maintenance](https://github.com/SubramaniMokkala/aircraft-predictive-maintenance)

---
