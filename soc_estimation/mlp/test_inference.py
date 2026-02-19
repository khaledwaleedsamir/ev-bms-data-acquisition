import pandas as pd
import glob
import numpy as np
from soc_estimation.mlp.mlp import MLP, ModelManager
from soc_estimation.dataset_manager import DatasetManager
from torchinfo import summary
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Path to downloaded dataset folder
data_path = r'D:\test_SVR\Battery-State-of-Charge-Estimation\dataset\LG_HG2_processed\25degC\*.csv'

# Load all CSV files
all_files = glob.glob(data_path)
df_list = [pd.read_csv(f) for f in all_files]
data = pd.concat(df_list, ignore_index=True)
print(data.head())

feature_cols = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Capacity [Ah]']
target_col = 'SOC [-]'

x = data[feature_cols].values
y = data[target_col].values.reshape(-1, 1)

# split into train and test sets (5% train, 95% test)
split_idx = int(0.90 * len(x))
x_train, y_train = x[:split_idx], y[:split_idx]
x_test, y_test = x[split_idx:], y[split_idx:]

# Load scalers and model
save_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\soc_estimation\mlp\outputs'

model = MLP(input_size=len(feature_cols), hidden_sizes=[64, 32, 16], output_size=1)
mlp_manager = ModelManager(model, device='cpu')
mlp_manager.load_model_weights(f"{save_path}\\best_mlp_model_combined.pth")

# Print model summary
summary(model, input_size=(1, len(feature_cols)))
mlp_manager.load_scalers(f"{save_path}\\scalers_combined_big.pkl")

# Warmup before inference loop
mlp_manager.model.eval()

# ── Real-time simulation loop 
predictions = np.zeros(len(x_test))  # pre-allocate instead of appending

for i in tqdm(range(len(x_test)), desc="Predicting", unit="sample"):
    sample = x_test[i]                           # shape (n_features,) — mimics one sensor reading
    predictions[i] = mlp_manager.predict(sample)[0]

# Results 
y_true = y_test.flatten()

mae  = mean_absolute_error(y_true, predictions)
rmse = np.sqrt(mean_squared_error(y_true, predictions))
r2   = r2_score(y_true, predictions)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot true vs predicted SOC
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True SOC', alpha=0.7)
plt.plot(predictions, label='Predicted SOC', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('State of Charge (SOC)')
plt.title('True vs Predicted SOC')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()