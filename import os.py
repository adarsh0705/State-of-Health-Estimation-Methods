import os
import pandas as pd
from glob import glob

# -------------------------------
# USER SETTINGS
# -------------------------------
bms_folder = r"C:\Users\DELL\Desktop\ML TRAIN_RUN\New folder"   # folder containing multiple BMS csvs
soh_curve_file = "battery SOH reference.csv"  # your experimental cycle vs SOH reference
output_file = "training_dataset.csv"          # final ML-ready dataset

# -------------------------------
# STEP 1: Load experimental SOH curve
# -------------------------------
# Load SOH vs cycle_count data
soh_df = pd.read_csv(soh_curve_file)

# Rename columns (your file had merged headers like "cycle_countBattery_health")
soh_df = soh_df.rename(columns={
    "cycle_countBattery_health": "cycle_count",
    "SOH": "soh"
})

# Ensure numeric values
soh_df["cycle_count"] = pd.to_numeric(soh_df["cycle_count"], errors="coerce").round().astype(int)
soh_df["soh"] = pd.to_numeric(soh_df["soh"], errors="coerce")

# Drop invalid rows
soh_df = soh_df.dropna(subset=["cycle_count", "soh"])

print("✅ Experimental SOH curve loaded and cleaned")
print(soh_df.head())

# -------------------------------
# STEP 2: Function to process a single BMS dataset
# -------------------------------
def process_bms_file(file_path, battery_id):
    """
    Reads one BMS dataset and aggregates features cycle-wise.
    """
    df = pd.read_csv(file_path)

    # Ensure 'cycle_count' column exists
    if "cycle_count" not in df.columns:
        raise ValueError(f"File {file_path} has no 'cycle_count' column")

    # Drop rows without cycle count
    df = df.dropna(subset=["cycle_count"])

    # Convert cycle_count to integer
    df["cycle_count"] = pd.to_numeric(df["cycle_count"], errors="coerce").dropna().astype(int)

    # Choose useful features (drop time/date columns)
    feature_cols = [col for col in df.columns if col not in ["time", "year", "month", "day"]]

    # Aggregate per cycle (mean, min, max, std)
    agg_funcs = ["mean", "min", "max", "std"]
    cycle_features = df.groupby("cycle_count")[feature_cols].agg(agg_funcs)

    # Flatten multi-index column names (e.g., voltage_mean, current_max, etc.)
    cycle_features.columns = ["_".join(col).strip() for col in cycle_features.columns.values]
    cycle_features.reset_index(inplace=True)

    # Add battery ID column (to differentiate multiple packs)
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

# Combine all processed packs into one dataset
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
