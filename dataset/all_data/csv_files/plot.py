import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

path = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test3  _pred.csv"
# Load CSV file
df = pd.read_csv(path)

# Extract columns
actual = df["Actual_SOC"]
predicted = df["Predicted_SOC"]

# ---- Metrics ----
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("===================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print("===================")

# ---- Line Plot ----
plt.figure()

plt.plot(actual.values, linestyle='-', label='Actual SOC')      # solid
plt.plot(predicted.values, linestyle='--', label='Predicted SOC')  # dotted
#limit y axis to 0-100%
# plt.ylim(0, 100)

plt.xlabel("Sample Index")
plt.ylabel("SOC")
plt.title("Actual vs Predicted SOC")
plt.legend()
plt.show()
