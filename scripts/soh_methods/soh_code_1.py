from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Loading dataset...")
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
file_path = DATA_DIR / "CAN_Data_Dump_For_PINV502949_30-Aug_15_08_23.csv"
df = pd.read_csv(file_path)

print("Columns available:", df.columns.tolist())
print("Total rows:", len(df))

print("Converting time to seconds...")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["time_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

df = df.dropna(subset=["time_sec"]).reset_index(drop=True)
print("Valid rows after cleaning:", len(df))


print("\nStep 2: Coulomb counting...")


Q_rated = df["battery capacity"].mode()[0]  # in Ah
print("Rated battery capacity (Ah):", Q_rated)

df["delta_t"] = df["time_sec"].diff().fillna(0)
df["Ah_change"] = (df["current( A )"] * df["delta_t"]) / 3600.0  # Ah


df["cum_Ah_discharge"] = df["Ah_change"].where(df["current( A )"] < 0, 0).cumsum().abs()

soc_low, soc_high = 20, 80
usable = df[(df["soc( % )"] >= soc_low) & (df["soc( % )"] <= soc_high)]
if not usable.empty:
    Q_measured = usable["cum_Ah_discharge"].iloc[-1] - usable["cum_Ah_discharge"].iloc[0]
else:
    Q_measured = np.nan

SOH_cap = Q_measured / Q_rated if Q_measured > 0 else np.nan
print(f"Estimated SOH (capacity-based): {SOH_cap:.3f}")

print("\nStep 3: Voltage–SOC method...")

cell_col = "cell voltage_01"
voltage_data = df[[cell_col, "soc( % )"]].dropna()


mid_range = voltage_data[(voltage_data["soc( % )"] >= 40) &
                         (voltage_data["soc( % )"] <= 60)]
V_obs = mid_range[cell_col].mean()

V_ref = 3.7
SOH_volt = V_obs / V_ref if V_ref > 0 else np.nan
print(f"Observed mid-SOC voltage: {V_obs:.3f} V")
print(f"Estimated SOH (voltage-based): {SOH_volt:.3f}")

print("\nStep 4: Resistance estimation...")

df["dV"] = df[cell_col].diff()
df["dI"] = df["current( A )"].diff()
df["R_est"] = df["dV"] / df["dI"]

R_measured = df["R_est"].replace([np.inf, -np.inf], np.nan).dropna().median()

R_new = 0.01
SOH_res = R_new / R_measured if R_measured > 0 else np.nan
print(f"Median estimated resistance: {R_measured:.5f} Ω")
print(f"Estimated SOH (resistance-based): {SOH_res:.3f}")

print("\nStep 5: Hybrid SOH fusion...")


w1, w2, w3 = 0.5, 0.3, 0.2


SOH_final = np.nanmean([w1*SOH_cap, w2*SOH_volt, w3*SOH_res])
print(f"\nFinal Estimated SOH: {SOH_final:.3f}")

plt.figure(figsize=(10,6))
plt.plot(df["time_sec"]/3600, df["soc( % )"], label="SOC (%)")
plt.plot(df["time_sec"]/3600, df[cell_col], label="Cell Voltage (V)")
plt.xlabel("Time (hours)")
plt.ylabel("Value")
plt.title("SOC & Voltage Trend")
plt.legend()
plt.grid(True)
plt.show()
