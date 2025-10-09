import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. Load & Inspect Data
# ================================
print("Loading dataset...")
file_path = "CAN_Data_Dump_For_PINV502949_30-Aug_15_03_48.csv"
df = pd.read_csv(file_path)

print("Columns available:", df.columns.tolist())
print("Total rows:", len(df))

# Convert time column → datetime
print("Converting time...")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).reset_index(drop=True)

# Time in seconds from start
df["time_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

# Extract day index (Day 1, Day 2, …)
df["day"] = (df["time"].dt.floor("D") - df["time"].iloc[0].floor("D")).dt.days + 1
print("Unique days in dataset:", df["day"].unique())

# ================================
# 2. Pre-calculations
# ================================
print("\nPrecomputing values...")

# Nominal capacity from dataset
Q_rated = df["battery capacity"].mode()[0]
print("Rated battery capacity (Ah):", Q_rated)

# Δt between samples
df["delta_t"] = df["time_sec"].diff().fillna(0)

# Ah change (positive charge, negative discharge)
df["Ah_change"] = (df["current( A )"] * df["delta_t"]) / 3600.0

# Cumulative discharged Ah
df["cum_Ah_discharge"] = df["Ah_change"].where(df["current( A )"] < 0, 0).cumsum().abs()

# For voltage-SOC method: pick one cell
cell_col = "cell voltage_01"
V_ref = 3.7      # typical mid-SOC voltage per cell

# ================================
# 3. SOH Estimation at Each Step
# ================================
print("\nEstimating SOH for each row...")

# Capacity-based SOH (cumulative discharge vs rated)
df["SOH_cap"] = (df["cum_Ah_discharge"] / Q_rated).clip(0, 1.2)   # fraction of new

# Voltage-based SOH (normalized to reference)
df["SOH_volt"] = (df[cell_col] / V_ref).clip(0, 1.2)

# Hybrid SOH (weighted fusion: 70% capacity, 30% voltage)
w1, w2 = 0.7, 0.3
df["SOH_final"] = (w1 * df["SOH_cap"] + w2 * df["SOH_volt"]).clip(0, 1.2)

print("SOH calculation complete for all rows.")

# ================================
# 4. Save to Excel
# ================================
output_path = "soh_estimates_1.xlsx"
df.to_excel(output_path, index=False)
print("Updated dataset with SOH saved to:", output_path)

# ================================
# 5. Plots
# ================================

# SOC vs Time
plt.figure(figsize=(10,5))
plt.plot(df["time_sec"]/3600, df["soc( % )"], label="SOC (%)")
plt.xlabel("Time (hours)")
plt.ylabel("SOC (%)")
plt.title("SOC Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Cell Voltage vs Time
plt.figure(figsize=(10,5))
plt.plot(df["time_sec"]/3600, df[cell_col], label=f"{cell_col}")
plt.xlabel("Time (hours)")
plt.ylabel("Cell Voltage (V)")
plt.title("Cell Voltage Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

# SOH Trends over Time
plt.figure(figsize=(10,6))
plt.plot(df["time_sec"]/3600, df["SOH_cap"], label="SOH - Coulomb Counting")
plt.plot(df["time_sec"]/3600, df["SOH_volt"], label="SOH - Voltage-SOC")
plt.plot(df["time_sec"]/3600, df["SOH_final"], label="SOH - Hybrid (Weighted)", linestyle="--")
plt.xlabel("Time (hours)")
plt.ylabel("SOH (fraction of new)")
plt.title("SOH Estimation Trends (Per Timestamp)")
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.show()
