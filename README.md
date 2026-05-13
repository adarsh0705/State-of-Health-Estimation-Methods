<div align="center">

# 🔋 Battery State-of-Health Estimation Methods

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/TensorFlow-DNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/PyBaMM-Physics%20Model-purple?style=for-the-badge"/>

<br/>

> **A comprehensive toolkit for estimating lithium-ion battery State-of-Health (SOH) using physics-based, signal-based, and machine-learning methods — applied to real CAN bus BMS data.**

</div>

---

## 📖 Table of Contents

- [🔍 Overview](#-overview)
- [🗂️ Repository Structure](#️-repository-structure)
- [⚙️ Methods](#️-methods)
  - [Method 1 — Single-pass Hybrid SOH](#method-1--single-pass-hybrid-soh)
  - [Method 2 — Daily SOH with 3-Method Fusion](#method-2--daily-soh-with-3-method-fusion)
  - [Method 3 — Daily SOH with 2-Method Fusion](#method-3--daily-soh-with-2-method-fusion)
  - [SOH CSV Update — Per-Timestamp Export](#soh-csv-update--per-timestamp-export)
  - [SOH Pipeline — PyBaMM Physics Calibration](#soh-pipeline--pybamm-physics-calibration)
  - [ML Pipeline — RF + XGBoost](#ml-pipeline--rf--xgboost)
  - [Full ML Pipeline — RF + XGBoost + DNN](#full-ml-pipeline--rf--xgboost--dnn)
- [📊 Results & Visualisations](#-results--visualisations)
- [📐 SOH Estimation Formula](#-soh-estimation-formula)
- [🛠️ Requirements](#️-requirements)
- [🚀 Quick Start](#-quick-start)
- [📁 Data Format](#-data-format)
- [🤝 Contributing](#-contributing)

---

## 🔍 Overview

**State-of-Health (SOH)** is a key indicator of a battery's remaining capacity and performance relative to its original specification. Accurate SOH estimation enables:

- 🔋 Proactive battery maintenance and replacement scheduling
- ⚡ Optimised range estimation in electric vehicles (EVs)
- 🛡️ Prevention of over-discharge and thermal runaway
- 📈 Data-driven battery lifecycle analytics

This repository implements **multiple SOH estimation strategies** — from simple physics-based heuristics to trained machine learning models — all designed to work with **real-world CAN bus data** captured from a Battery Management System (BMS).

---

## 🗂️ Repository Structure

```
State-of-Health-Estimation-Methods/
│
├── 📄 SOH code 1.py              # Single-pass hybrid SOH (Coulomb + Voltage + Resistance)
├── 📄 SOH code 2.py              # Daily SOH — 3-method fusion (50/30/20 weights)
├── 📄 SOH code 3.py              # Daily SOH — 2-method fusion (70/30 weights)
├── 📄 SOH csv update.py          # Per-timestamp SOH computation + Excel export
├── 📄 soh pipeline.py            # PyBaMM SPMe physics model + R_series calibration
├── 📄 Train and Run ML RF XGB.py # ML pipeline: Random Forest + XGBoost
├── 📄 train plus ML.py           # Lightweight ML pipeline variant
├── 📄 TRAIN RF XG DNN.py         # Full ML pipeline: RF + XGBoost + DNN (Keras)
│
├── 🖼️ SOH code 1 SOV and Voltage plots.png
├── 🖼️ SOH code 1 results.PNG
├── 🖼️ SOH code 2 SOC curve.png
├── 🖼️ SOH code 2 SOH plot.png
├── 🖼️ SOH code 2 Voltage curve.png
├── 🖼️ SOH code 2 soh comparison.png
├── 🖼️ SOH code 3 Results.PNG
├── 🖼️ SOH code 3 SOC curve.png
├── 🖼️ SOH code 3 SOC overall SOH plot.png
├── 🖼️ SOH code 3 SOH combined plot.png
└── 🖼️ SOH code 3 Voltage curve.png
```

---

## ⚙️ Methods

### Method 1 — Single-pass Hybrid SOH

**File:** `SOH code 1.py`

A quick, single-pass SOH estimate fusing three independent physics-based signals:

| Signal | Approach | Weight |
|--------|----------|--------|
| ⚡ Capacity | Coulomb counting (SOC 20–80%) | 50% |
| 🔌 Voltage | Mid-SOC voltage vs reference | 30% |
| 🔩 Resistance | dV/dI ratio vs new-cell resistance | 20% |

```
SOH_final = 0.5 × SOH_cap + 0.3 × SOH_volt + 0.2 × SOH_res
```

**Output:** Single printed SOH estimate + SOC & voltage trend plots.

---

### Method 2 — Daily SOH with 3-Method Fusion

**File:** `SOH code 2.py`

Extends Method 1 to compute SOH **per calendar day**, enabling degradation trend tracking over time.

- Groups data by day (`Day 1`, `Day 2`, …)
- Applies the same 3-method hybrid (50 / 30 / 20)
- Generates per-day SOH comparison plots

**Outputs:**
- SOC trend over time
- Cell voltage trend over time
- Daily hybrid SOH trend
- Side-by-side comparison of all 3 SOH signals

---

### Method 3 — Daily SOH with 2-Method Fusion

**File:** `SOH code 3.py`

A streamlined daily estimator using only **Coulomb counting** and **Voltage-SOC**, dropping resistance (useful when dV/dI is noisy):

```
SOH_final = 0.7 × SOH_cap + 0.3 × SOH_volt
```

**Outputs:** Same plots as Method 2, optimised for cleaner resistance-free datasets.

---

### SOH CSV Update — Per-Timestamp Export

**File:** `SOH csv update.py`

Computes SOH at **every timestep** (not just daily) and saves the enriched dataset to Excel (`.xlsx`). Ideal for downstream ML training or detailed analysis.

- Adds `SOH_cap`, `SOH_volt`, and `SOH_final` columns row-by-row
- Exports full dataframe to `soh_estimates_1.xlsx`

---

### SOH Pipeline — PyBaMM Physics Calibration

**File:** `soh pipeline.py`

Uses **PyBaMM** to build a physics-based electrochemical simulation (SPMe model with lumped thermal) and fits an **external series resistance** `R_series` so that the simulated terminal voltage matches the measured pack voltage.

```
V_simulated(t) + I(t) × R_series  ≈  V_measured(t)
```

- Model: `pybamm.lithium_ion.SPMe` with `"thermal": "lumped"`
- Parameter set: `LGM50_Gravimetric`
- Optimiser: `scipy.optimize.least_squares` (20 iterations)
- Outputs: `fit_trace.csv`, `fitted_params.json`, voltage fit + residual plots

---

### ML Pipeline — RF + XGBoost

**File:** `Train and Run ML RF XGB.py` / `train plus ML.py`

A supervised machine learning pipeline that maps **per-cycle BMS features → SOH**, trained against an experimental SOH reference curve.

**Feature engineering per charge cycle:**

| Category | Features |
|----------|----------|
| Voltage | mean, min, max, std, range, slope proxy |
| Current | mean, min, max, std, abs mean, charge/discharge mean |
| Power | mean, max |
| Temperature | range per cell |
| Cell imbalance | mean voltage diff, std across cells |
| Capacity / SOC | mean available capacity, mean SOC |

**Models:**
- 🌲 **Random Forest** — 200 trees, max depth 10
- 🚀 **XGBoost** — 200 estimators, learning rate 0.1

**Metrics:** R² Score, RMSE, Feature Importances (top-10)

---

### Full ML Pipeline — RF + XGBoost + DNN

**File:** `TRAIN RF XG DNN.py`

Extends the ML pipeline with **delta features** (cycle-over-cycle changes) and adds a **Deep Neural Network**:

```
Input → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(1)
```

- Optimiser: Adam (lr=0.001), Loss: MSE
- Trained for 50 epochs, batch size 32
- Scaled with `StandardScaler`
- Generates actual vs predicted scatter plots for all 3 models

---

## 📊 Results & Visualisations

<table>
  <tr>
    <td align="center"><b>SOC & Voltage Trend (Code 1)</b></td>
    <td align="center"><b>SOH Comparison (Code 2)</b></td>
  </tr>
  <tr>
    <td><img src="SOH code 1 SOV and Voltage plots.png" width="400"/></td>
    <td><img src="SOH code 2 soh comparison.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>SOC Curve (Code 3)</b></td>
    <td align="center"><b>Combined SOH Plot (Code 3)</b></td>
  </tr>
  <tr>
    <td><img src="SOH code 3 SOC curve.png" width="400"/></td>
    <td><img src="SOH code 3 SOH combined plot.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Voltage Curve (Code 3)</b></td>
    <td align="center"><b>Overall SOH Plot (Code 3)</b></td>
  </tr>
  <tr>
    <td><img src="SOH code 3 Voltage curve.png" width="400"/></td>
    <td><img src="SOH code 3 SOC overall SOH plot.png" width="400"/></td>
  </tr>
</table>

---

## 📐 SOH Estimation Formula

The **hybrid SOH** used throughout this project is a weighted combination of independent estimators:

$$\text{SOH}_{\text{final}} = w_1 \cdot \text{SOH}_{\text{cap}} + w_2 \cdot \text{SOH}_{\text{volt}} + w_3 \cdot \text{SOH}_{\text{res}}$$

Where:

| Term | Formula | Description |
|------|---------|-------------|
| $\text{SOH}_{\text{cap}}$ | $Q_{\text{measured}} / Q_{\text{rated}}$ | Ratio of delivered Ah to rated capacity |
| $\text{SOH}_{\text{volt}}$ | $V_{\text{obs}} / V_{\text{ref}}$ | Mid-SOC cell voltage vs reference |
| $\text{SOH}_{\text{res}}$ | $R_{\text{new}} / R_{\text{measured}}$ | New-cell resistance vs current resistance |

> Weights are tunable: Code 1 & 2 use `(0.5, 0.3, 0.2)`, Code 3 uses `(0.7, 0.3)`.

---

## 🛠️ Requirements

Install all dependencies with:

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost tensorflow pybamm openpyxl
```

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
| `scipy` | Optimisation (least squares) |
| `scikit-learn` | Random Forest, preprocessing, metrics |
| `xgboost` | Gradient-boosted trees |
| `tensorflow` / `keras` | Deep Neural Network |
| `pybamm` | Physics-based electrochemical simulation |
| `openpyxl` | Excel export |

> **Python version:** 3.8 or higher recommended.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/adarsh0705/State-of-Health-Estimation-Methods.git
cd State-of-Health-Estimation-Methods
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost tensorflow pybamm openpyxl
```

### 3. Prepare your data

Place your CAN bus CSV file (see [Data Format](#-data-format) below) in the project folder and update the `file_path` variable at the top of whichever script you want to run.

### 4. Run a script

```bash
# Quick single-pass SOH estimate
python "SOH code 1.py"

# Daily SOH trend (3-method fusion)
python "SOH code 2.py"

# Daily SOH trend (2-method fusion, resistance-free)
python "SOH code 3.py"

# Export per-timestamp SOH to Excel
python "SOH csv update.py"

# Train ML models (RF + XGBoost + DNN)
python "TRAIN RF XG DNN.py"
```

---

## 📁 Data Format

All scripts expect a **CSV file** exported from a CAN bus BMS logger. The required columns are:

| Column | Description |
|--------|-------------|
| `time` | Timestamp (parseable by `pd.to_datetime`) |
| `current( A )` | Pack current in Amperes (negative = discharge) |
| `soc( % )` | State-of-Charge in percent |
| `battery capacity` | Rated capacity in Ah |
| `cell voltage_01` | Cell #1 voltage in Volts |

> Additional cell voltage columns (`cell voltage_02`, …) and temperature columns are used by the ML scripts for richer feature extraction.

The ML pipeline additionally requires a **SOH reference CSV** (`battery SOH reference.csv`) with columns:

| Column | Description |
|--------|-------------|
| `cycle_countBattery_health` | Charge-cycle index |
| `SOH` | Ground-truth SOH (0–1) |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or pull request for:

- 🔧 Additional SOH estimation methods (EKF, Particle Filter, LSTM)
- 🧪 Hyperparameter tuning scripts
- 📦 Packaging the pipeline as a reusable Python module
- 📚 Jupyter notebook versions of each script

---

<div align="center">

Made with ❤️ for battery health research and EV diagnostics.

</div>
