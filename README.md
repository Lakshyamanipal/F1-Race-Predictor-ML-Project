F1 Race Predictor — ML Project 

Machine-learning pipeline to predict **F1 race podium finishes** based on driver skill, qualifying position, constructor performance, weather conditions, tyre strategy, and rolling historical statistics.

--- 

## 📁 Project Layout 

```python 
f1_predictor/ 
├── f1_predictor.py ← Data → Features → Train model → Predict 
├── f1_eda.py ← EDA & visualization plots 
├── requirements.txt ← Python dependencies 
└── README.md ← You are here! 
``` 

--- 

## 🚀 Getting Started 

```bash 
# 1. Install dependencies 
pip install -r requirements.txt 

# 2. Run the main pipeline to train model & make predictions
python f1_predictor.py 

# 3. (Optional) Make EDA plots from `f1_eda.py` 
python f1_eda.py 
``` 

--- 

## 🏗️ Pipeline Stages 

### 1. Data Generation / Loading 

- Synthetic dataset of 500 races with 10 drivers each
- Swap out `generate_f1_dataset()` to use live data: 
- [`Ergast API`](#ergast-api) for race timing results 
- [`FastF1`](#fastf1-python-package) for live telemetry & timing 

### 2. Feature Engineering 

| Feature | Description | 
|----------------|-----------------------------------------| 
| `grid_position`| Driver's qualifying start position | 
| `driver_skill` | Driver's skill rating (normalized) | 
| `car_performance` | Constructor's car rating | 
| `avg_finish_last5` | Driver's average finishing position for last 5 races |
| `podium_rate_last10` | Driver's podium percentage for last 10 races |
| `weather_enc` | One-hot encoded weather condition | 
| `circuit_enc` | One-hot encoded race circuit | 
| `tyre_enc` | One-hot encoded tyre strategy | 
| `pit_stops` | Planned number of pit stops | 
| `dnf` | Binary flag indicating if driver DNF'd | 

### 3. Models 

- **Gradient Boosting** (GBM) classifier — primary model 
- **Random Forest** classifier — benchmark model 
- Use 5-fold cross-validation to evaluate both models 
- Automatically select best model 

### 4. Prediction 

- Accepts 3 arguments: `circuit`, `weather`, `season` 
- Returns ranked probability of drivers finishing on podium

--- 

## 🔌 Replaceable Data Sources 

Replace `generate_f1_dataset()` with live data instead: 


```python 
# Option A: Ergast REST API (free) 
>>> import requests 
>>> url = "http://ergast.com/api/f1/2024/results.json?limit=500" 
>>> data = requests.get(url).json() 

# Option B: FastF1 Python package (telemetry + timing) 
>>> import fastf1 
>>> session = fastf1.get_session(2024, 'Monza', 'R') 
>>> session.load() 
>>> laps = session.laps 
``` 

--- 

### 📊 Example Terminal Output 

``` 
============================ 
F1 RACE PREDICTOR — MODEL TRAINING 
============================ 
[GBM] Test Accuracy : 0.8340 
[GBM] CV Accuracy : 0.8212 ± 0.0143 
✅ Best Model: Gradient Boosting 

🏁 RACE PREDICTION: MONZA (2025) 
Pos Driver Grid Podium Prob 
────────────────────────────────────────── 
1 Verstappen 3 74.2% ██████████████ 
2 Leclerc 1 68.5% █████████████ 
3 Hamilton 2 61.3% ████████████ 
... 
``` 
