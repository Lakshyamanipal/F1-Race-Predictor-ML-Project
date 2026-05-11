"""
F1 Race Predictor — EDA & Visualization
========================================
Run this after f1_predictor.py to explore the dataset and model results.

Requires: matplotlib, seaborn
    pip install matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Re-use data generation from main module ────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from f1_predictor import generate_f1_dataset, engineer_features


# ─────────────────────────────────────────────
#  PLOT 1 — Podium Rate by Driver
# ─────────────────────────────────────────────

def plot_driver_podium_rates(df: pd.DataFrame):
    podium_rates = (
        df.groupby("driver")["podium"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    podium_rates.columns = ["Driver", "Podium Rate"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e10600" if i < 3 else "#1f77b4" for i in range(len(podium_rates))]
    bars = ax.barh(podium_rates["Driver"], podium_rates["Podium Rate"],
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Podium Rate", fontsize=12)
    ax.set_title("🏆 Podium Rate by Driver", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 0.7)
    for bar, val in zip(bars, podium_rates["Podium Rate"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("f1_podium_rates.png", dpi=150)
    print("  Saved: f1_podium_rates.png")
    plt.show()


# ─────────────────────────────────────────────
#  PLOT 2 — Grid Position vs Finish Position
# ─────────────────────────────────────────────

def plot_grid_vs_finish(df: pd.DataFrame):
    avg = (
        df[df["dnf"] == 0]
        .groupby("grid_position")["finish_position"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(avg["grid_position"], avg["finish_position"],
            marker="o", color="#e10600", linewidth=2.5, markersize=8)
    ax.fill_between(avg["grid_position"], avg["finish_position"],
                    alpha=0.15, color="#e10600")
    ax.set_xlabel("Grid Position (Qualifying)", fontsize=12)
    ax.set_ylabel("Avg Finish Position", fontsize=12)
    ax.set_title("Grid → Finish: Starting Position Effect", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("f1_grid_vs_finish.png", dpi=150)
    print("  Saved: f1_grid_vs_finish.png")
    plt.show()


# ─────────────────────────────────────────────
#  PLOT 3 — Constructor Win Share (Heatmap)
# ─────────────────────────────────────────────

def plot_circuit_constructor_heatmap(df: pd.DataFrame):
    pivot = (
        df[df["finish_position"] == 1]
        .groupby(["circuit", "constructor"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds",
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("🏎️ Race Wins by Circuit × Constructor", fontsize=14, fontweight="bold")
    ax.set_xlabel("Constructor", fontsize=11)
    ax.set_ylabel("Circuit", fontsize=11)
    plt.tight_layout()
    plt.savefig("f1_circuit_wins_heatmap.png", dpi=150)
    print("  Saved: f1_circuit_wins_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────
#  PLOT 4 — Weather Effect on Positions
# ─────────────────────────────────────────────

def plot_weather_effect(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    df[df["dnf"] == 0].boxplot(
        column="finish_position", by="weather", ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", color="white"),
        medianprops=dict(color="#e10600", linewidth=2.5),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )
    ax.set_title("Weather Conditions vs Finish Position Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Weather", fontsize=12)
    ax.set_ylabel("Finish Position", fontsize=12)
    ax.invert_yaxis()
    plt.suptitle("")  # remove default title
    plt.tight_layout()
    plt.savefig("f1_weather_effect.png", dpi=150)
    print("  Saved: f1_weather_effect.png")
    plt.show()


# ─────────────────────────────────────────────
#  PLOT 5 — Season Performance Trend
# ─────────────────────────────────────────────

def plot_season_trend(df: pd.DataFrame):
    trend = (
        df.groupby(["season", "driver"])["race_points"]
        .sum()
        .reset_index()
    )
    top_drivers = (
        trend.groupby("driver")["race_points"]
        .sum()
        .nlargest(5)
        .index
    )
    trend = trend[trend["driver"].isin(top_drivers)]

    palette = ["#e10600", "#00D2BE", "#DC0000", "#FF8000", "#006EFF"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, driver in enumerate(top_drivers):
        data = trend[trend["driver"] == driver].sort_values("season")
        ax.plot(data["season"], data["race_points"],
                marker="o", label=driver, linewidth=2.5,
                color=palette[i % len(palette)])

    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Total Points", fontsize=12)
    ax.set_title("📈 Top 5 Driver Points Trend by Season", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("f1_season_trend.png", dpi=150)
    print("  Saved: f1_season_trend.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("  Loading dataset for EDA ...")
    raw_df = generate_f1_dataset(n_races=500)
    df = engineer_features(raw_df)

    print("  Plotting Driver Podium Rates ...")
    plot_driver_podium_rates(df)

    print("  Plotting Grid vs Finish ...")
    plot_grid_vs_finish(df)

    print("  Plotting Circuit × Constructor Heatmap ...")
    plot_circuit_constructor_heatmap(df)

    print("  Plotting Weather Effect ...")
    plot_weather_effect(df)

    print("  Plotting Season Trend ...")
    plot_season_trend(df)

    print("\n  ✅ All plots saved.")
