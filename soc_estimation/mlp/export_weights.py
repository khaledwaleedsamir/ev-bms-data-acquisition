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
print("------------------------------------------------")
print("StandardScaler Parameters: ")
print("Mean: ", mlp_manager.scaler_X.mean_)
print("Scale: ",mlp_manager.scaler_X.scale_)
print("------------------------------------------------")
print("Model Params: ")
params = {}
for name, param in mlp_manager.model.named_parameters():
    params[name] = param.detach().cpu().numpy()

for k, v in params.items():
    print(k, v.shape)

# print each layer's weights and biases
for name, param in mlp_manager.model.named_parameters():
    print(f"Layer: {name}")
    print(param.detach().cpu().numpy())
    print("------------------------------------------------")


print("Exporting weights in C format:")

def export_c(name, arr):
    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        print(f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{")
        for row in arr:
            print("    {" + ", ".join(f"{x:.6f}f" for x in row) + "},")
        print("};\n")
    else:
        print(f"const float {name}[{arr.shape[0]}] = {{")
        print("    " + ", ".join(f"{x:.6f}f" for x in arr))
        print("};\n")

mapping = {
    "network.0.weight": "W1",
    "network.0.bias": "b1",
    "network.3.weight": "W2",
    "network.3.bias": "b2",
    "network.6.weight": "W3",
    "network.6.bias": "b3",
}

for name, param in model.named_parameters():
    export_c(mapping[name], param.detach().cpu().numpy())