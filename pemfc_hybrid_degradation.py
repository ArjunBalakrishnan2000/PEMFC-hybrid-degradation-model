# importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path

# Loading the data

DATA_PATH = Path(__file__).resolve().parent / "data"

fc_part1 = pd.read_excel(DATA_PATH / "FC_Part1.xlsx", decimal=",")
fc_part2 = pd.read_excel(DATA_PATH / "FC_Part2.xlsx", decimal=",")
fc_init = pd.concat([fc_part1, fc_part2], ignore_index=True)
fc_init["Time (h)"] = pd.to_numeric(fc_init["Time (h)"], errors="coerce")
fc_init.index = pd.to_timedelta(fc_init["Time (h)"], unit="h")

df = fc_init.sort_index()
df = df[~df.index.duplicated(keep="first")]
df_hourly = df.resample("1h").mean().dropna()
df_hourly["hours_elapsed"] = df_hourly.index.total_seconds() / 3600.0
fc = df_hourly.reset_index(drop=True)

N_CELLS = 5
SPLIT = 800.0

TIMESTAMPS_HR = [0, 35, 182, 343, 515, 666]
EIS_CURRENTS = [20, 45, 70]

pola_dict = {
    hr: pd.read_excel(
        DATA_PATH / f"FC2_Pola_T{hr:03d}.xlsx",
        decimal=","
    )
    for hr in TIMESTAMPS_HR
}
eis_dict = {
    (hr, cur): pd.read_excel(
        DATA_PATH / f"FC2_EIS{cur}A_postpola_T{hr:03d}.xlsx",
        decimal=","
    )
    for hr in TIMESTAMPS_HR
    for cur in EIS_CURRENTS
}

# Calculating the ohmic resistance using impedance spectroscopy data

def get_hfr_from_eis(eis_df):
    df_clean = eis_df.dropna(subset=["fREQUENCY/hZ", "i/oHM", "r/oHM"])
    df_clean = df_clean.sort_values("fREQUENCY/hZ", ascending=False).reset_index(drop=True)
    z_imag = df_clean["i/oHM"].to_numpy()
    z_real = df_clean["r/oHM"].to_numpy()

    sign_change = np.where(np.diff(np.sign(z_imag)) != 0)[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        x0, x1 = z_imag[i], z_imag[i + 1]
        y0, y1 = z_real[i], z_real[i + 1]
        return float(y0 + (0 - x0) * (y1 - y0) / (x1 - x0)) if x1 != x0 else float(y0)
    return float(z_real[np.argmin(np.abs(z_imag))])


b_stack_list = []
for hr in TIMESTAMPS_HR:
    hfr_vals = [get_hfr_from_eis(eis_dict[(hr, cur)]) for cur in EIS_CURRENTS]
    b_stack_list.append(np.mean(hfr_vals))

fc["R_ohmic_stack"] = np.interp(fc["hours_elapsed"], TIMESTAMPS_HR, b_stack_list)
fc["R_ohmic_cell"] = fc["R_ohmic_stack"] / N_CELLS

# Calculating the optimal circuit voltage, activation loss factor and mass transport factor

e_ocv_list, a_factor, c_factor = [], [], []

for hr, R_ohmic_stack in zip(TIMESTAMPS_HR, b_stack_list):
    df_pola = pola_dict[hr].dropna(subset=["I (A)", "Ustack (V)"])

    idx_min_I = df_pola["I (A)"].idxmin()
    e_ocv_moment = df_pola.loc[idx_min_I, "Ustack (V)"] / N_CELLS
    e_ocv_list.append(e_ocv_moment)

    I = df_pola["I (A)"].to_numpy()
    U_stack_actual = df_pola["Ustack (V)"].to_numpy()
    V_cell_actual = U_stack_actual / N_CELLS
    R_cell = R_ohmic_stack / N_CELLS

    mask = I > 0.5
    I_fit = I[mask]
    V_cell_fit = V_cell_actual[mask]
    y_fit = e_ocv_moment - V_cell_fit - (I_fit * R_cell)
    X = np.column_stack([np.log(I_fit), np.sqrt(I_fit)])

    res = lsq_linear(X, y_fit, bounds=([1e-5, 0.0], [np.inf, np.inf]))
    a_factor.append(res.x[0])
    c_factor.append(res.x[1])
  
#  Calculating the residual
fc["E_cell_ocv"] = np.interp(fc["hours_elapsed"], TIMESTAMPS_HR, e_ocv_list)
fc["a_param"] = np.interp(fc["hours_elapsed"], TIMESTAMPS_HR, a_factor)
fc["c_param"] = np.interp(fc["hours_elapsed"], TIMESTAMPS_HR, c_factor)

fc["V_cell_pred"] = (
    fc["E_cell_ocv"]
    - fc["a_param"] * np.log(fc["I (A)"].clip(lower=0.5))
    - fc["R_ohmic_cell"] * fc["I (A)"]
    - fc["c_param"] * np.sqrt(fc["I (A)"].clip(lower=0))
)
fc["Ustack_pred"] = fc["V_cell_pred"] * N_CELLS
fc["Ustack_residual"] = fc["Utot (V)"] - fc["Ustack_pred"]

dt_hr = fc["hours_elapsed"].diff().fillna(0).clip(lower=0)
fc["cumulative_Ah"] = (fc["I (A)"] * dt_hr).cumsum()

# Splitting the data
train = fc[fc["hours_elapsed"] <= SPLIT].copy()
test = fc[fc["hours_elapsed"] > SPLIT].copy()

# Calculating the degradation trend
train["residual_smooth"] = (
    train["Ustack_residual"].rolling(window=51, center=True, min_periods=20).median()
)
train["residual_smooth"] = train["residual_smooth"].interpolate().bfill().ffill()

X_deg_train = train[["hours_elapsed", "cumulative_Ah"]].values
y_deg_train = train["residual_smooth"].values

degradation_model = Ridge(alpha=0.1)
degradation_model.fit(X_deg_train, y_deg_train)

train["r_degradation"] = degradation_model.predict(X_deg_train)
test["r_degradation"] = degradation_model.predict(test[["hours_elapsed", "cumulative_Ah"]].values)

# Calculating the fluctuations in the degradation
OPERATING_FEATURES = ["I (A)", "TinH2 (ｰC)", "ToutH2 (ｰC)", "DoutAIR (l/mn)", "R_ohmic_cell"]

train["r_operating_target"] = train["Ustack_residual"] - train["r_degradation"]

fluctuation_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, reg_alpha=0.1,
    objective="reg:squarederror", random_state=42,
)
fluctuation_model.fit(train[OPERATING_FEATURES], train["r_operating_target"])

train["r_operating_pred"] = fluctuation_model.predict(train[OPERATING_FEATURES])
test["r_operating_pred"] = fluctuation_model.predict(test[OPERATING_FEATURES])

# final prediction
test["r_predicted"] = test["r_degradation"] + test["r_operating_pred"]
test["U_hybrid"] = test["Ustack_pred"] + test["r_predicted"]

# Evaluation
physics_rmse = np.sqrt(mean_squared_error(test["Utot (V)"], test["Ustack_pred"]))
physics_mae = mean_absolute_error(test["Utot (V)"], test["Ustack_pred"])
physics_r2 = r2_score(test["Utot (V)"], test["Ustack_pred"])

hybrid_rmse = np.sqrt(mean_squared_error(test["Utot (V)"], test["U_hybrid"]))
hybrid_mae = mean_absolute_error(test["Utot (V)"], test["U_hybrid"])
hybrid_r2 = r2_score(test["Utot (V)"], test["U_hybrid"])

print("=" * 60)
print("Physics model only:")
print(f"RMSE = {physics_rmse*1000:.2f} mV, MAE = {physics_mae*1000:.2f} mV, R2 = {physics_r2:.4f}")
print("\nPhysics + Degradation + XGBoost:")
print(f"RMSE = {hybrid_rmse*1000:.2f} mV, MAE = {hybrid_mae*1000:.2f} mV, R2 = {hybrid_r2:.4f}")
print(f"\nRMSE reduction = {(1 - hybrid_rmse/physics_rmse)*100:.2f}%")
print("=" * 60)

# Visualisation
plt.figure(figsize=(14, 7))
plt.plot(test["hours_elapsed"], test["Utot (V)"], linewidth=2, label="Measured voltage")
plt.plot(test["hours_elapsed"], test["Ustack_pred"], linewidth=1.5, linestyle="--", label="Physics model")
plt.plot(test["hours_elapsed"], test["U_hybrid"], linewidth=2, label="Hybrid PIML")
plt.axvline(SPLIT, linestyle=":", linewidth=2, label="Train/Test split")
plt.xlabel("Operating hours")
plt.ylabel("Stack voltage (V)")
plt.title("PEMFC Hybrid Physics-Informed Residual Model")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
