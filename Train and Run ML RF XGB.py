import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# -------------------------------
# USER SETTINGS
# -------------------------------
bms_folder = r"C:\Users\DELL\Desktop\ML TRAIN_RUN\New folder"
soh_curve_file = "battery SOH reference.csv"
output_file = "training_dataset.csv"

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
# STEP 2: Function to process a single BMS dataset
# -------------------------------
def process_bms_file(file_path, battery_id):
    df = pd.read_csv(file_path)
    if "cycle_count" not in df.columns:
        raise ValueError(f"File {file_path} has no 'cycle_count' column")
    
    df = df.dropna(subset=["cycle_count"])
    df["cycle_count"] = pd.to_numeric(df["cycle_count"], errors="coerce").dropna().astype(int)

    # Identify useful features
    feature_cols = [col for col in df.columns if col not in ["time", "year", "month", "day"]]

    # -------------------------------
    # Base Aggregations (mean, min, max, std)
    # -------------------------------
    agg_funcs = ["mean", "min", "max", "std"]
    cycle_features = df.groupby("cycle_count")[feature_cols].agg(agg_funcs)
    cycle_features.columns = ["_".join(col).strip() for col in cycle_features.columns.values]

    # -------------------------------
    # Extra Engineered Features
    # -------------------------------
    # Voltage-based
    if "voltage" in df.columns:
        v_group = df.groupby("cycle_count")["voltage"]
        cycle_features["voltage_range"] = v_group.max() - v_group.min()
        cycle_features["voltage_slope_proxy"] = v_group.apply(lambda x: (x.max() - x.min()) / max(len(x), 1))
    
    # Current-based
    if "current" in df.columns:
        i_group = df.groupby("cycle_count")["current"]
        cycle_features["current_abs_mean"] = i_group.apply(lambda x: x.abs().mean())
        cycle_features["charge_current_mean"] = i_group.apply(lambda x: x[x > 0].mean() if any(x > 0) else 0)
        cycle_features["discharge_current_mean"] = i_group.apply(lambda x: x[x < 0].mean() if any(x < 0) else 0)

    # Power-based
    if {"voltage", "current"}.issubset(df.columns):
        df["power"] = df["voltage"] * df["current"]
        p_group = df.groupby("cycle_count")["power"]
        cycle_features["power_mean"] = p_group.mean()
        cycle_features["power_max"] = p_group.max()

    # Temperature-based
    temp_cols = [c for c in df.columns if "temp" in c.lower()]
    for tcol in temp_cols:
        t_group = df.groupby("cycle_count")[tcol]
        cycle_features[f"{tcol}_range"] = t_group.max() - t_group.min()

    # Cell imbalance (if cell voltage columns exist)
    cell_cols = [c for c in df.columns if "cell" in c.lower() and "voltage" in c.lower()]
    if len(cell_cols) > 1:
        def imbalance_fn(x):
            return x.max(axis=1).mean() - x.min(axis=1).mean()
        def imbalance_std_fn(x):
            return x.std(axis=1).mean()
        cycle_features["cell_voltage_diff_mean"] = df.groupby("cycle_count")[cell_cols].apply(imbalance_fn)
        cycle_features["cell_voltage_std_mean"] = df.groupby("cycle_count")[cell_cols].apply(imbalance_std_fn)

    cycle_features.reset_index(inplace=True)
    cycle_features["battery_id"] = battery_id
    return cycle_features

# -------------------------------
# STEP 3: Process all BMS CSVs in folder
# -------------------------------
all_files = glob(os.path.join(bms_folder, "*.csv"))
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
# STEP 5: Save ML-ready dataset
# -------------------------------
training_df.to_csv(output_file, index=False)
print(f"\n✅ Training dataset saved to: {output_file}")

# -------------------------------
# STEP 6: Train & Evaluate ML Models
# -------------------------------
X = training_df.drop(columns=["soh", "battery_id", "cycle_count"])
y = training_df["soh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

def print_feature_importance(model, X, model_name):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": X.columns, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    print(f"\n🔥 {model_name} Feature Importances:")
    print(fi_df.head(10))

print_feature_importance(rf_model, X, "Random Forest")
print_feature_importance(xgb_model, X, "XGBoost")
