import torch
from soc_estimation.mlp.mlp import MLP, ModelManager
from soc_estimation.dataset_manager import DatasetManager
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
from torchinfo import summary
import numpy as np

class TensorPairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataset path
data_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\hoverboard_bms_dataset_combined.h5'
# Output model and scalar save path
save_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\soc_estimation\mlp\outputs'

try:
    with h5py.File(data_path, 'r') as f:
        # List all groups
        print("Keys in the h5 file:", list(f.keys()))
except Exception as e:
    print(f"Error loading file: {e}")

raw_data_list = []

# Open the HDF5 file
with h5py.File(data_path, 'r') as f:
    for run_name in f.keys():
        run_group = f[run_name]

        # Access bms group and extract specific datasets as numpy arrays
        bms_group = run_group['bms']
        battery_level = bms_group['battery_level'][:]
        temp_values = bms_group['temp_values'][:]
        voltage = bms_group['voltage'][:]
        current = bms_group['current'][:]
        power = bms_group['power'][:]
        cycle_capacity = bms_group['cycle_capacity'][:]
        cycle_charge = bms_group['cycle_charge'][:]

        # Extract 'timestamp_ms' from the run group top level
        timestamp_ms = run_group['timestamp_ms'][:]

        # Store extracted arrays and metadata in a dictionary
        run_data = {
            'run_name': run_name,
            'battery_level': battery_level,
            'temp_values': temp_values,
            'voltage': voltage,
            'current': current,
            'power': power,
            'cycle_capacity': cycle_capacity,
            'cycle_charge': cycle_charge,
            'timestamp_ms': timestamp_ms,
            'metadata': dict(run_group.attrs)
        }

        raw_data_list.append(run_data)

print(f"Successfully extracted data from {len(raw_data_list)} runs.")
run_dfs = []
for run in raw_data_list:
    # Create DataFrame for current run with scalar series
    df = pd.DataFrame({
        'timestamp_ms': run['timestamp_ms'],
        'SOC [-]': run['battery_level']/100,
        'Voltage [V]': run['voltage'],
        'Current [A]': run['current'],
        'power': run['power'],
        'cycle_capacity': run['cycle_capacity'],
        'Capacity [Ah]': run['cycle_charge']
    })

    # Add run_name column
    df['run_name'] = run['run_name']

    # Flatten temp_values into temp_1, temp_2, temp_3
    temp_vals = run['temp_values']
    df['temp_1'] = temp_vals[:, 0]
    df['temp_2'] = temp_vals[:, 1]
    df['temp_3'] = temp_vals[:, 2]

    run_dfs.append(df)

# Concatenate all runs into one master DataFrame
bms_df = pd.concat(run_dfs, ignore_index=True)

# Display summary info
print(f"Unified DataFrame shape: {bms_df.shape}")
# Add average temperature column
bms_df['Temperature [degC]'] = bms_df[['temp_1', 'temp_2', 'temp_3']].mean(axis=1)
# Check head of the DataFrame
print(bms_df.head())

train_runs = [
    'file1_run_001',
    'file1_run_002',
    'file1_run_003',
    'file1_run_004',
    'file1_run_005',
    'file2_run_001_40pct_speed_15kg_load_discharge',
    'file2_run_002_40pct_speed_15kg_load_discharge',
    'file2_run_003_charge',
    'file2_run_004_80pct_speed_15kg_load_discharge',
    'file2_run_005_charge',
    'file2_run_006_60pct_speed_15kg_load_discharge',
    'file2_run_009_40pct_speed_25kg_load_discharge',
    'file2_run_010_80pct_speed_25kg_load_discharge',
    'file2_run_011_80pct_speed_25kg_load_discharge',
    'file2_run_012_80pct_speed_25kg_load_discharge',
    'file2_run_013_80pct_speed_25kg_load_discharge',
    'file2_run_014_charge'
]

val_runs = [
    'file2_run_007_60pct_speed_15kg_load_discharge',
    'file2_run_008_charge'
]

train_df = bms_df[bms_df.run_name.isin(train_runs)].copy()
val_df   = bms_df[bms_df.run_name.isin(val_runs)].copy()

print(f"Train samples: {len(train_df)}")
print(f"Val samples:   {len(val_df)}")

feature_cols = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Capacity [Ah]']
target_col = 'SOC [-]'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values.reshape(-1, 1)

X_val = val_df[feature_cols].values
y_val = val_df[target_col].values.reshape(-1, 1)

print("Length of train data: ", len(X_train))
print("Length of val data: ", len(X_val))

num_ip_features = len(feature_cols)

# Normalize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f"{save_path}\\final_scalers.pkl")
    
# Create datasets and dataloaders
train_dataset = TensorPairDataset(X_train_scaled, y_train_scaled)
val_dataset = TensorPairDataset(X_val_scaled, y_val_scaled)

# Create MLP model
device = 'cpu'
model = MLP(input_size=num_ip_features, hidden_sizes=[64, 32, 16], output_size=1)

# print model summary
summary(model, input_size=(1, num_ip_features))

# Create Model Manager
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
mlp_manager = ModelManager(model, device=device, optimizer=optimizer, criterion=criterion)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

history = mlp_manager.start_training(train_loader=train_loader, val_loader=val_loader, epochs=100, patience=20, save_path=f"{save_path}\\final_best_mlp_model.pth", verbose=True)

# plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training History')
plt.legend()
plt.grid()
plt.show()