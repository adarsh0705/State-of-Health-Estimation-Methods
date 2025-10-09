import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading dataset...")
file_path = "CAN_Data_Dump_For_PINV502949_30-Aug_15_08_23.csv"
df = pd.read_csv(file_path)

print("Columns available:", df.columns.tolist())
print("Total rows:", len(df))

print("Converting time...")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).reset_index(drop=True)

df["time_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

df["day"] = (df["time"].dt.floor("D") - df["time"].iloc[0].floor("D")).dt.days + 1
print("Unique days in dataset:", df["day"].unique())

print("\nPrecomputing values...")

Q_rated = df["battery capacity"].mode()[0]
print("Rated battery capacity (Ah):", Q_rated)

df["delta_t"] = df["time_sec"].diff().fillna(0)

df["Ah_change"] = (df["current( A )"] * df["delta_t"]) / 3600.0

df["cum_Ah_discharge"] = df["Ah_change"].where(df["current( A )"] < 0, 0).cumsum().abs()

cell_col = "cell voltage_01"
df["dV"] = df[cell_col].diff()
df["dI"] = df["current( A )"].diff()
df["R_est"] = df["dV"] / df["dI"]

V_ref = 3.7      
R_new = 0.25    

print("\nEstimating SOH per day...")

w1, w2, w3 = 0.5, 0.3, 0.2  

day_list = []
soh_cap_list = []
soh_volt_list = []
soh_res_list = []
soh_final_list = []

for day in sorted(df["day"].unique()):
    chunk = df[df["day"] == day]
    if len(chunk) < 100:
        continue
    
    usable = chunk[(chunk["soc( % )"] >= 20) & (chunk["soc( % )"] <= 80)]
    if not usable.empty:
        Q_measured = usable["cum_Ah_discharge"].iloc[-1] - usable["cum_Ah_discharge"].iloc[0]
        SOH_cap = Q_measured / Q_rated if Q_measured > 0 else np.nan
    else:
        SOH_cap = np.nan
    
    mid_range = chunk[(chunk["soc( % )"] >= 40) & (chunk["soc( % )"] <= 60)]
    V_obs = mid_range[cell_col].mean() if not mid_range.empty else np.nan
    SOH_volt = V_obs / V_ref if V_obs > 0 else np.nan
    
    R_measured = chunk["R_est"].replace([np.inf, -np.inf], np.nan).dropna().median()
    SOH_res = R_new / R_measured if (R_measured and R_measured > 0) else np.nan
    
    SOH_final = np.nanmean([w1*SOH_cap, w2*SOH_volt, w3*SOH_res])
    
    print(f"Day {day} → SOH_cap={SOH_cap:.3f}, SOH_volt={SOH_volt:.3f}, SOH_res={SOH_res:.3f}, Final={SOH_final:.3f}")
    
    day_list.append(day)
    soh_cap_list.append(SOH_cap)
    soh_volt_list.append(SOH_volt)
    soh_res_list.append(SOH_res)
    soh_final_list.append(SOH_final)

plt.figure(figsize=(10,5))
plt.plot(df["time_sec"]/3600, df["soc( % )"], label="SOC (%)")
plt.xlabel("Time (hours)")
plt.ylabel("SOC (%)")
plt.title("SOC Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["time_sec"]/3600, df[cell_col], label=f"{cell_col}")
plt.xlabel("Time (hours)")
plt.ylabel("Cell Voltage (V)")
plt.title("Cell Voltage Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(day_list, soh_final_list, marker="o", linestyle="-", color="b", label="Hybrid SOH")
plt.xlabel("Day")
plt.ylabel("SOH (fraction of new)")
plt.title("Battery SOH Trend (Daily Estimate)")
plt.xticks(day_list)
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(day_list, soh_cap_list, marker="o", label="SOH - Coulomb Counting")
plt.plot(day_list, soh_volt_list, marker="s", label="SOH - Voltage-SOC")
plt.plot(day_list, soh_res_list, marker="^", label="SOH - Resistance")
plt.plot(day_list, soh_final_list, marker="d", linestyle="--", label="SOH - Hybrid (Weighted)")
plt.xlabel("Day")
plt.ylabel("SOH (fraction of new)")
plt.title("SOH Estimation Comparison (Daily)")
plt.xticks(day_list)
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.show()
