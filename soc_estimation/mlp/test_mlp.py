import torch
from soc_estimation.mlp.mlp import MLP_SOC, ModelManager
from soc_estimation.dataset_manager import DatasetManager
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
from torchinfo import summary
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Output model and scalar save path
save_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\soc_estimation\mlp\outputs'
model = MLP_SOC(input_size=5, hidden_sizes=[32, 16], output_size=1)
mlp_manager = ModelManager(model, device='cpu')
mlp_manager.load_model_weights(f"{save_path}\\mlp_model2.pth")
mlp_manager.load_scalers(f"{save_path}\\scalers2.pkl")
mlp_manager.model.eval()
# Print model summary
summary(mlp_manager.model, input_size=(1, 5))

# Load the dataset
hdf5_file = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\h5_files\hoverboard_bms_dataset.h5"
run_name = "/run_002"

bms_samples = []
all_true_soc = []
all_predicted_soc = []
all_input_features = []

with h5py.File(hdf5_file, "r") as f:
    g_run = f[run_name]["bms"]  

    # get dataset lengths
    n_samples = g_run["battery_level"].shape[0]

    # iterate sample by sample
    for i in range(n_samples):
        sample = {
            "battery_charging": bool(g_run["battery_charging"][i]),
            "battery_level": float(g_run["battery_level"][i]),
            "voltage": float(g_run["voltage"][i]),
            "current": float(g_run["current"][i]),
            "cycle_charge": (g_run["cycle_charge"][i]),
            "temp_sensors": int(g_run["temp_sensors"][i]),
            "temp_values": g_run["temp_values"][i].tolist(),
            "power": float(g_run["power"][i]),
            "cycle_capacity": float(g_run["cycle_capacity"][i]),
            "cycles": float(g_run["cycles"][i]),
            "delta_voltage": float(g_run["delta_voltage"][i]),
            "temperature": float(g_run["temperature"][i]),
            "cell_count": int(g_run["cell_count"][i]),
            "cell_voltages": g_run["cell_voltages"][i].tolist()
        }
        bms_samples.append(sample)
        data_for_mlp = [
            sample.get("voltage", 0.0),
            sample.get("current", 0.0),
            np.mean(sample.get("temp_values", [0,0,0])),
            sample.get("cycle_charge", 0),
            sample.get("cycle_capacity", 0.0)
        ]
        predicted_soc = mlp_manager.predict(data_for_mlp)[0] * 100  # Scale back to percentage
        all_predicted_soc.append(predicted_soc)
        all_true_soc.append(sample.get("battery_level", 0.0))
        all_input_features.append(data_for_mlp)



# calculate metrics
mae = mean_absolute_error(all_true_soc, all_predicted_soc)
mse = mean_squared_error(all_true_soc, all_predicted_soc)
r2 = r2_score(all_true_soc, all_predicted_soc)
print(f"MAE: {mae:.2f}%")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot true vs predicted SOC
import matplotlib.pyplot as plt
# Set figure size and style
plt.figure(figsize=(10, 6))

# Plot true vs predicted SOC
plt.plot(all_true_soc, label='True SOC', linestyle='-', color='blue', linewidth=1)
plt.plot(all_predicted_soc, label='Predicted SOC', linestyle='--', color='red', linewidth=1)

# Titles and labels
plt.title('True vs Predicted SOC', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('SOC (%)', fontsize=14)

# Legend
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()  # adds minor ticks

# Optional: tighten layout for papers
plt.tight_layout()

# Show plot
plt.show()

input_features_df = pd.DataFrame(all_input_features, columns=[
    "Voltage",
    "Current",
    "Temp_Mean",
    "Cycle_Charge",
    "Cycle_Capacity"
])
df_results = pd.DataFrame({
    "Actual_SOC": all_true_soc,
    "Predicted_SOC": all_predicted_soc
})
df_results = pd.concat([df_results, input_features_df], axis=1)
df_results.to_csv(r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test6_pred.csv", index=False)

