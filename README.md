# F1 Race Predictor — ML Project

A machine-learning pipeline to predict **F1 race podium finishes** using
driver skill, qualifying position, constructor performance, weather, tyre strategy,
and rolling historical stats.

---

## 📁 Project Structure

```
f1_predictor/
├── f1_predictor.py   ← Main pipeline (data → features → train → predict)
├── f1_eda.py         ← EDA & visualization plots
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model and run predictions
python f1_predictor.py

# 3. (Optional) Run EDA visualizations
python f1_eda.py
```

---

## 🏗️ Pipeline Overview

### 1. Data Generation / Loading
- Synthetic dataset (500 races × 10 drivers)
- Swap `generate_f1_dataset()` with real data from the **Ergast API** or **FastF1**

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `grid_position` | Qualifying start position |
| `driver_skill` | Normalized driver rating |
| `car_performance` | Constructor car rating |
| `avg_finish_last5` | Rolling 5-race average finish |
| `podium_rate_last10` | Rolling 10-race podium % |
| `weather_enc` | Encoded weather condition |
| `circuit_enc` | Encoded circuit |
| `tyre_enc` | Encoded tyre strategy |
| `pit_stops` | Number of pit stops |
| `dnf` | Binary: Did Not Finish flag |

### 3. Models
- **Gradient Boosting** (GBM) — primary model
- **Random Forest** — ensemble comparison
- 5-fold cross-validation for both
- Best model selected automatically

### 4. Prediction
- Takes `circuit`, `weather`, `season` as inputs
- Outputs ranked podium probability for all drivers

---

## 🔌 Real Data Sources

Replace the synthetic generator with live data:

```python
# Option A: Ergast REST API (free)
import requests
url = "http://ergast.com/api/f1/2024/results.json?limit=500"
data = requests.get(url).json()

# Option B: FastF1 (telemetry + timing)
import fastf1
session = fastf1.get_session(2024, 'Monza', 'R')
session.load()
laps = session.laps
```

---

## 📊 Expected Output

```
============================
  F1 RACE PREDICTOR — MODEL TRAINING
============================
  [GBM]  Test Accuracy : 0.8340
  [GBM]  CV Accuracy   : 0.8212 ± 0.0143
  ✅ Best Model: Gradient Boosting

  🏁 RACE PREDICTION: MONZA (2025)
  Pos   Driver          Grid     Podium Prob
  ──────────────────────────────────────────
  1     Verstappen      3        74.2%  ██████████████
  2     Leclerc         1        68.5%  █████████████
  3     Hamilton        2        61.3%  ████████████
  ...
```

---

## 🛠️ Extend the Project

- [ ] Add **qualifying lap time** as a feature
- [ ] Include **championship standings** pressure
- [ ] Use **SHAP** for explainability
- [ ] Build a **Streamlit** dashboard for interactive predictions
- [ ] Add **hyperparameter tuning** with Optuna
