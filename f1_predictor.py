"""
F1 Race Predictor - Main Pipeline
==================================
Predicts F1 race finishing positions using historical race data,
qualifying results, driver standings, and constructor performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. SYNTHETIC DATA GENERATOR
#  (Replace with real Ergast / FastF1 API data)
# ─────────────────────────────────────────────

def generate_f1_dataset(n_races: int = 500) -> pd.DataFrame:
    """
    Generates a realistic synthetic F1 dataset.
    Each row = one driver's result in one race.
    """
    np.random.seed(42)

    drivers = [
        "Verstappen", "Hamilton", "Leclerc", "Norris", "Sainz",
        "Russell", "Alonso", "Perez", "Piastri", "Stroll"
    ]
    constructors = [
        "Red Bull", "Mercedes", "Ferrari", "McLaren", "Ferrari",
        "Mercedes", "Aston Martin", "Red Bull", "McLaren", "Aston Martin"
    ]
    driver_constructor = dict(zip(drivers, constructors))

    # Base skill ratings (0-1 scale)
    driver_skill = {
        "Verstappen": 0.97, "Hamilton": 0.95, "Leclerc": 0.91,
        "Norris": 0.89, "Sainz": 0.87, "Russell": 0.86,
        "Alonso": 0.90, "Perez": 0.84, "Piastri": 0.83, "Stroll": 0.75
    }
    constructor_performance = {
        "Red Bull": 0.96, "Mercedes": 0.88, "Ferrari": 0.87,
        "McLaren": 0.89, "Aston Martin": 0.78
    }

    circuits = ["Monza", "Spa", "Monaco", "Silverstone", "Suzuka",
                "Bahrain", "Singapore", "Austin", "Brazil", "Abu Dhabi"]

    records = []
    for race_id in range(n_races):
        circuit = np.random.choice(circuits)
        season = np.random.randint(2018, 2025)
        weather = np.random.choice(["Dry", "Wet", "Mixed"], p=[0.65, 0.20, 0.15])

        # Shuffle grid for qualifying simulation
        grid_order = drivers.copy()
        np.random.shuffle(grid_order)

        for grid_pos, driver in enumerate(grid_order, 1):
            constructor = driver_constructor[driver]
            skill = driver_skill[driver]
            car_perf = constructor_performance[constructor]

            # Base score combining driver skill + car
            base_score = (skill * 0.55) + (car_perf * 0.35) + np.random.normal(0, 0.08)

            # Grid position penalty
            grid_penalty = (grid_pos - 1) * 0.015
            base_score -= grid_penalty

            # Weather randomness (wet = more chaos)
            if weather == "Wet":
                base_score += np.random.normal(0, 0.12)
            elif weather == "Mixed":
                base_score += np.random.normal(0, 0.07)

            # Circuit-specific boosts
            if circuit == "Monaco" and driver in ["Leclerc", "Alonso"]:
                base_score += 0.05
            if circuit == "Monza" and constructor in ["Ferrari"]:
                base_score += 0.04

            # Mechanical failure / safety car randomness
            dnf = np.random.random() < 0.06  # 6% DNF chance
            pit_stops = np.random.randint(1, 4)
            tyre_strategy = np.random.choice(["Soft-Medium", "Medium-Hard", "Soft-Hard", "One-Stop"])

            records.append({
                "race_id": race_id,
                "season": season,
                "circuit": circuit,
                "weather": weather,
                "driver": driver,
                "constructor": constructor,
                "grid_position": grid_pos,
                "driver_skill": skill,
                "car_performance": car_perf,
                "base_score": base_score,
                "dnf": int(dnf),
                "pit_stops": pit_stops,
                "tyre_strategy": tyre_strategy,
            })

    df = pd.DataFrame(records)

    # Derive finish position from base_score (higher score = better finish)
    df["finish_position"] = (
        df.groupby("race_id")["base_score"]
        .rank(ascending=False)
        .astype(int)
    )

    # If DNF → push to back
    df.loc[df["dnf"] == 1, "finish_position"] = (
        df.loc[df["dnf"] == 1, "finish_position"] + 7
    )
    df["finish_position"] = df["finish_position"].clip(1, 10)

    # Target: podium (top 3)
    df["podium"] = (df["finish_position"] <= 3).astype(int)

    return df


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates model-ready features from raw race data."""

    df = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    df["driver_enc"]       = le.fit_transform(df["driver"])
    df["constructor_enc"]  = le.fit_transform(df["constructor"])
    df["circuit_enc"]      = le.fit_transform(df["circuit"])
    df["weather_enc"]      = le.fit_transform(df["weather"])
    df["tyre_enc"]         = le.fit_transform(df["tyre_strategy"])

    # Rolling average finish position per driver (last 5 races)
    df = df.sort_values(["driver", "race_id"])
    df["avg_finish_last5"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Rolling podium rate per driver (last 10 races)
    df["podium_rate_last10"] = (
        df.groupby("driver")["podium"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Constructor rolling average finish
    df["constructor_avg_finish"] = (
        df.groupby("constructor")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Grid position squared (non-linear effect)
    df["grid_sq"] = df["grid_position"] ** 2

    # Front row bonus
    df["front_row"] = (df["grid_position"] <= 2).astype(int)

    # Points in race (F1 scoring)
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                  6:  8, 7:  6, 8:  4, 9:  2, 10: 1}
    df["race_points"] = df["finish_position"].map(points_map).fillna(0)

    df["avg_finish_last5"].fillna(df["finish_position"].mean(), inplace=True)
    df["podium_rate_last10"].fillna(0.2, inplace=True)
    df["constructor_avg_finish"].fillna(5.0, inplace=True)

    return df


# ─────────────────────────────────────────────
#  3. MODEL TRAINING
# ─────────────────────────────────────────────

FEATURES = [
    "grid_position", "grid_sq", "front_row",
    "driver_skill", "car_performance",
    "weather_enc", "circuit_enc",
    "driver_enc", "constructor_enc", "tyre_enc",
    "avg_finish_last5", "podium_rate_last10",
    "constructor_avg_finish", "pit_stops", "dnf",
    "season"
]

def train_models(df: pd.DataFrame):
    """Trains GBM + Random Forest for podium prediction."""

    X = df[FEATURES]
    y = df["podium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("=" * 60)
    print("  F1 RACE PREDICTOR — MODEL TRAINING")
    print("=" * 60)
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"  Podium rate      : {y.mean():.1%}")
    print()

    # ── Gradient Boosting ──────────────────────
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, random_state=42
    )
    gbm.fit(X_train_sc, y_train)
    gbm_preds = gbm.predict(X_test_sc)
    gbm_acc   = accuracy_score(y_test, gbm_preds)

    cv_gbm = cross_val_score(gbm, X_train_sc, y_train, cv=5, scoring="accuracy")
    print(f"  [GBM]  Test Accuracy : {gbm_acc:.4f}")
    print(f"  [GBM]  CV Accuracy   : {cv_gbm.mean():.4f} ± {cv_gbm.std():.4f}")

    # ── Random Forest ──────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    rf_preds = rf.predict(X_test_sc)
    rf_acc   = accuracy_score(y_test, rf_preds)

    cv_rf = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring="accuracy")
    print(f"\n  [RF]   Test Accuracy : {rf_acc:.4f}")
    print(f"  [RF]   CV Accuracy   : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # Pick best model
    best_model = gbm if gbm_acc >= rf_acc else rf
    best_name  = "Gradient Boosting" if gbm_acc >= rf_acc else "Random Forest"
    best_preds = gbm_preds if gbm_acc >= rf_acc else rf_preds

    print(f"\n  ✅ Best Model: {best_name}")
    print()
    print("  Classification Report (Best Model):")
    print(classification_report(y_test, best_preds,
          target_names=["No Podium", "Podium"]))

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
    print("  Top 8 Feature Importances:")
    print(importances.nlargest(8).to_string())
    print()

    return best_model, scaler


# ─────────────────────────────────────────────
#  4. RACE PREDICTOR
# ─────────────────────────────────────────────

def predict_race(model, scaler, df: pd.DataFrame,
                 circuit: str, weather: str, season: int):
    """
    Predicts podium probabilities for a new race weekend.

    Parameters
    ----------
    model   : trained sklearn model
    scaler  : fitted StandardScaler
    df      : full engineered DataFrame (for rolling stats lookup)
    circuit : circuit name (must exist in training data)
    weather : "Dry" | "Wet" | "Mixed"
    season  : race season year
    """

    drivers = [
        "Verstappen", "Hamilton", "Leclerc", "Norris", "Sainz",
        "Russell", "Alonso", "Perez", "Piastri", "Stroll"
    ]
    constructors = [
        "Red Bull", "Mercedes", "Ferrari", "McLaren", "Ferrari",
        "Mercedes", "Aston Martin", "Red Bull", "McLaren", "Aston Martin"
    ]
    driver_skill = {
        "Verstappen": 0.97, "Hamilton": 0.95, "Leclerc": 0.91,
        "Norris": 0.89, "Sainz": 0.87, "Russell": 0.86,
        "Alonso": 0.90, "Perez": 0.84, "Piastri": 0.83, "Stroll": 0.75
    }
    constructor_perf = {
        "Red Bull": 0.96, "Mercedes": 0.88, "Ferrari": 0.87,
        "McLaren": 0.89, "Aston Martin": 0.78
    }

    le_circuit = LabelEncoder().fit(df["circuit"])
    le_weather = LabelEncoder().fit(df["weather"])
    le_driver  = LabelEncoder().fit(df["driver"])
    le_constr  = LabelEncoder().fit(df["constructor"])
    le_tyre    = LabelEncoder().fit(df["tyre_strategy"])

    # Simulated qualifying grid (random for demo)
    np.random.seed(99)
    grid = list(range(1, len(drivers) + 1))
    np.random.shuffle(grid)

    rows = []
    for i, (driver, constructor) in enumerate(zip(drivers, constructors)):
        grid_pos = grid[i]

        # Pull rolling stats from historical data
        drv_history = df[df["driver"] == driver].tail(10)
        avg_finish  = drv_history["finish_position"].mean() if len(drv_history) > 0 else 5.0
        podium_rate = drv_history["podium"].mean() if len(drv_history) > 0 else 0.2
        con_history = df[df["constructor"] == constructor].tail(5)
        con_avg     = con_history["finish_position"].mean() if len(con_history) > 0 else 5.0

        try:
            circuit_enc = le_circuit.transform([circuit])[0]
        except ValueError:
            circuit_enc = 0
        try:
            weather_enc = le_weather.transform([weather])[0]
        except ValueError:
            weather_enc = 0

        rows.append({
            "grid_position"        : grid_pos,
            "grid_sq"              : grid_pos ** 2,
            "front_row"            : int(grid_pos <= 2),
            "driver_skill"         : driver_skill[driver],
            "car_performance"      : constructor_perf[constructor],
            "weather_enc"          : weather_enc,
            "circuit_enc"          : circuit_enc,
            "driver_enc"           : le_driver.transform([driver])[0],
            "constructor_enc"      : le_constr.transform([constructor])[0],
            "tyre_enc"             : le_tyre.transform(["Medium-Hard"])[0],
            "avg_finish_last5"     : avg_finish,
            "podium_rate_last10"   : podium_rate,
            "constructor_avg_finish": con_avg,
            "pit_stops"            : 2,
            "dnf"                  : 0,
            "season"               : season,
            "_driver"              : driver,
            "_grid"                : grid_pos,
        })

    race_df = pd.DataFrame(rows)
    X_pred  = race_df[FEATURES]
    X_sc    = scaler.transform(X_pred)

    probs = model.predict_proba(X_sc)[:, 1]
    race_df["podium_probability"] = probs

    results = (
        race_df[["_driver", "_grid", "podium_probability"]]
        .rename(columns={"_driver": "Driver", "_grid": "Grid"})
        .sort_values("podium_probability", ascending=False)
        .reset_index(drop=True)
    )
    results.index += 1

    print("=" * 60)
    print(f"  🏁  RACE PREDICTION: {circuit.upper()}  ({season})")
    print(f"  Weather: {weather}")
    print("=" * 60)
    print(f"  {'Pos':<5} {'Driver':<15} {'Grid':<8} {'Podium Prob'}")
    print("  " + "-" * 40)
    for pos, row in results.iterrows():
        bar = "█" * int(row["podium_probability"] * 20)
        print(f"  {pos:<5} {row['Driver']:<15} {row['Grid']:<8} "
              f"{row['podium_probability']:.1%}  {bar}")
    print()
    print(f"  🏆 Predicted Winner : {results.iloc[0]['Driver']}")
    print(f"  🥈 Predicted P2    : {results.iloc[1]['Driver']}")
    print(f"  🥉 Predicted P3    : {results.iloc[2]['Driver']}")
    print("=" * 60)

    return results


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Generating F1 dataset ...")
    raw_df = generate_f1_dataset(n_races=500)

    print("  Engineering features ...")
    df = engineer_features(raw_df)

    print("  Training models ...\n")
    model, scaler = train_models(df)

    # ── Predict a new race ──────────────────────
    predict_race(
        model, scaler, df,
        circuit="Monza",
        weather="Dry",
        season=2025
    )

    predict_race(
        model, scaler, df,
        circuit="Monaco",
        weather="Wet",
        season=2025
    )
