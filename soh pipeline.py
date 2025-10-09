"""
pybamm_calibration_vs.py
Run in Visual Studio Python environment.

Steps:
1. Reads your CAN preprocessed CSV file.
2. Builds a PyBaMM SPMe model with lumped thermal.
3. Fits an external series resistance (R_series) so that
   simulated voltage matches measured pack voltage.
4. Saves results and plots.

Requirements (install these in VS terminal):
    pip install pybamm numpy scipy pandas matplotlib
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ===== USER SETTINGS =====
INPUT_CSV = r"CAN_Data_Dump_For_PINV502949_30-Aug_15_03_48.csv"   # change path
OUTPUT_DIR = r"C:\Users\DELL\Desktop\Fresh Start"       # results folder
PARAM_SET = "LGM50_Gravimetric"   # PyBaMM chemistry set
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Load data
# --------------------------
df = pd.read_csv(INPUT_CSV)

required_cols = ["time", "current", "battery_voltage"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in {INPUT_CSV}")

t = df["time_s"].values.astype(float)
I = df["current_A"].values.astype(float)
V = df["battery_voltage_V"].values.astype(float)

# --------------------------
# Import PyBaMM
# --------------------------
import pybamm

print("Loading PyBaMM parameter set:", PARAM_SET)
param = pybamm.ParameterValues(chemistry=getattr(pybamm.parameter_sets.lithium_ion, PARAM_SET))

# Build SPMe model
model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})

# Interpolant for current (positive = discharge)
current_interp = pybamm.Interpolant(t, I, pybamm.t, fill_value="extrapolate")
param.update({"Current function [A]": current_interp}, check_already_exists=False)

# Simulation setup
solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

t_span = [t[0], t[-1]]

# --------------------------
# Helper: simulate voltage with R_series
# --------------------------
def simulate_with_R(R_series):
    try:
        sol = sim.solve(t_span)
        Vsim = sol["Terminal voltage [V]"](t)
    except Exception as e:
        print("Simulation failed:", e)
        return np.full_like(V, 1e6)
    return np.array(Vsim) + I * R_series

# --------------------------
# Residual function for fitting
# --------------------------
def residual(theta):
    R_series = np.exp(theta[0])  # ensure positivity
    Vfit = simulate_with_R(R_series)
    return Vfit - V

# --------------------------
# Run optimization
# --------------------------
theta0 = [np.log(1e-3)]  # start ~1 mOhm
res = least_squares(residual, theta0, bounds=([np.log(1e-6)], [np.log(1.0)]), max_nfev=20)

R_series_fit = np.exp(res.x[0])
print("Fitted R_series [Ohm]:", R_series_fit)

# --------------------------
# Final simulation with fitted R
# --------------------------
Vfit = simulate_with_R(R_series_fit)

# --------------------------
# Save results
# --------------------------
df_out = pd.DataFrame({
    "time_s": t,
    "I_A": I,
    "V_meas_V": V,
    "V_fit_V": Vfit,
    "residual_V": Vfit - V
})
df_out.to_csv(os.path.join(OUTPUT_DIR, "fit_trace.csv"), index=False)

results = {
    "R_series_Ohm": float(R_series_fit),
    "success": bool(res.success),
    "message": res.message
}
with open(os.path.join(OUTPUT_DIR, "fitted_params.json"), "w") as f:
    json.dump(results, f, indent=2)

# --------------------------
# Plots
# --------------------------
plt.figure(figsize=(8,4))
plt.plot(t, V, label="Measured V")
plt.plot(t, Vfit, label=f"Fitted V (R={R_series_fit:.4f} Ω)")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "voltage_fit.png"), dpi=150)
plt.show()

plt.figure(figsize=(8,3))
plt.plot(t, Vfit - V)
plt.xlabel("Time [s]"); plt.ylabel("Residual (V)")
plt.title("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals.png"), dpi=150)
plt.show()

print("All outputs saved to:", OUTPUT_DIR)
