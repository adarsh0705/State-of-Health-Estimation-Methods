from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# USER SETTINGS
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

bms_folder = DATA_DIR / "bms"  # folder with BMS csv files
soh_curve_file = DATA_DIR / "battery_soh_reference.csv"  # your experimental SOH curve
output_file = OUTPUT_DIR / "training_dataset.csv"  # final ML-ready dataset

# -------------------------------
# STEP 1: Load experimental SOH curve
# -------------------------------
soh_df = pd.read_csv(soh_curve_file)
soh_df = soh_df.rename(columns={
    "cycle_countBattery_health": "cycle_count",
    "SOH": "soh"
})
soh_df["cycle_count"] = pd.to_numeric(soh_df["cycle_count"], errors="coerce").round().astype(int)
soh_df["soh"] = pd.to_numeric(soh_df["soh"], errors="coerce")
soh_df = soh_df.dropna(subset=["cycle_count", "soh"])
print("✅ Experimental SOH curve loaded and cleaned")
print(soh_df.head())

# -------------------------------
# STEP 2: Process a single BMS dataset
# -------------------------------
def process_bms_file(file_path, battery_id):
    df = pd.read_csv(file_path)

    if "cycle_count" not in df.columns:
        raise ValueError(f"File {file_path} has no 'cycle_count' column")

    df = df.dropna(subset=["cycle_count"])
    df["cycle_count"] = pd.to_numeric(df["cycle_count"], errors="coerce").dropna().astype(int)

    exclude_cols = ["time", "year", "month", "day"]
    num_cols = [c for c in df.columns if c not in exclude_cols]

    current_col = next((c for c in df.columns if c.lower().strip() == "current"), None)
    voltage_col = next((c for c in df.columns if "battery_voltage" in c.lower() or c.lower().strip() == "voltage"), None)
    cell_voltage_cols = [c for c in df.columns if "cell_voltage" in c.lower()]
    cell_temp_cols = [c for c in df.columns if "cell_temperature" in c.lower()]
    soc_col = next((c for c in df.columns if c.lower() == "soc"), None)
    capacity_col = next((c for c in df.columns if "available_capacity" in c.lower()), None)

    agg_funcs = ["mean", "min", "max", "std"]
    cycle_features = df.groupby("cycle_count")[num_cols].agg(agg_funcs)
    cycle_features.columns = ["_".join(col).strip() for col in cycle_features.columns.values]

    if voltage_col:
        v_group = df.groupby("cycle_count")[voltage_col]
        cycle_features["voltage_range"] = v_group.max() - v_group.min()
        cycle_features["voltage_slope_proxy"] = v_group.apply(lambda x: (x.max() - x.min())/max(len(x),1))

    if current_col:
        i_group = df.groupby("cycle_count")[current_col]
        cycle_features["current_abs_mean"] = i_group.apply(lambda x: x.abs().mean())
        cycle_features["charge_current_mean"] = i_group.apply(lambda x: x[x>0].mean() if any(x>0) else 0)
        cycle_features["discharge_current_mean"] = i_group.apply(lambda x: x[x<0].mean() if any(x<0) else 0)

    if current_col and voltage_col:
        df["power"] = df[voltage_col] * df[current_col]
        p_group = df.groupby("cycle_count")["power"]
        cycle_features["power_mean"] = p_group.mean()
        cycle_features["power_max"] = p_group.max()

    if len(cell_temp_cols) > 0:
        t_group = df.groupby("cycle_count")[cell_temp_cols]
        cycle_features["cell_temp_max"] = t_group.max().max(axis=1)
        cycle_features["cell_temp_min"] = t_group.min().min(axis=1)
        cycle_features["cell_temp_range"] = cycle_features["cell_temp_max"] - cycle_features["cell_temp_min"]

    if len(cell_voltage_cols) > 1:
        def imbalance_fn(x):
            return x.max(axis=1).mean() - x.min(axis=1).mean()
        def imbalance_std_fn(x):
            return x.std(axis=1).mean()
        cycle_features["cell_voltage_diff_mean"] = df.groupby("cycle_count")[cell_voltage_cols].apply(imbalance_fn)
        cycle_features["cell_voltage_std_mean"] = df.groupby("cycle_count")[cell_voltage_cols].apply(imbalance_std_fn)

    if soc_col:
        cycle_features["soc_mean"] = df.groupby("cycle_count")[soc_col].mean()
    if capacity_col:
        cycle_features["capacity_mean"] = df.groupby("cycle_count")[capacity_col].mean()

    cycle_features.reset_index(inplace=True)
    cycle_features["battery_id"] = battery_id
    return cycle_features

# -------------------------------
# STEP 3: Process all BMS CSVs
# -------------------------------
all_files = glob(str(bms_folder / "*.csv"))
all_data = []
for i, file in enumerate(all_files):
    print(f"Processing {file} ...")
    battery_id = f"battery_{i+1}"
    try:
        bms_processed = process_bms_file(file, battery_id)
        all_data.append(bms_processed)
    except Exception as e:
        print(f"⚠️ Skipping {file} due to error: {e}")

if len(all_data) == 0:
    raise RuntimeError("No valid BMS files processed!")

combined_df = pd.concat(all_data, ignore_index=True)
print("✅ Combined BMS data prepared")
print(combined_df.head())

# -------------------------------
# STEP 4: Merge with experimental SOH curve
# -------------------------------
training_df = pd.merge(combined_df, soh_df, on="cycle_count", how="inner")
print("✅ Training dataset merged with SOH curve")
print(training_df.head())

# -------------------------------
# STEP 5: Compute delta features
# -------------------------------
training_df = training_df.sort_values(["battery_id", "cycle_count"]).reset_index(drop=True)
delta_df = training_df.copy()
num_cols = delta_df.drop(columns=["battery_id", "cycle_count", "soh"]).select_dtypes(include=[np.number]).columns
for col in num_cols:
    delta_df[f"{col}_delta"] = delta_df.groupby("battery_id")[col].diff()

# -------------------------------
# STEP 6: Save ML-ready dataset
# -------------------------------
delta_df.to_csv(output_file, index=False)
print(f"\n✅ Training dataset with delta features saved to: {output_file}")

# -------------------------------
# STEP 7: Train & Evaluate Models
# -------------------------------
X = delta_df.drop(columns=["soh", "battery_id", "cycle_count"])
y = delta_df["soh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return rmse, r2

rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")
rmse_xgb, r2_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")

# -------------------------------
# STEP 8: Train & Evaluate a simple DNN
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history = model.fit(X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    verbose=1)

y_pred_dnn = model.predict(X_test_scaled).flatten()
rmse_dnn = np.sqrt(mean_squared_error(y_test, y_pred_dnn))
r2_dnn = r2_score(y_test, y_pred_dnn)
print(f"\n📊 DNN Evaluation:\nR² Score: {r2_dnn:.4f}\nRMSE: {rmse_dnn:.4f}")

# -------------------------------
# STEP 9: Plot predicted vs actual for each model
# -------------------------------
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"{model_name}: Actual vs Predicted SOH")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_soh_plot.png", dpi=150)
    plt.show()

plot_predictions(y_test, y_pred_rf, "RandomForest")
plot_predictions(y_test, y_pred_xgb, "XGBoost")
plot_predictions(y_test, y_pred_dnn, "DNN")
