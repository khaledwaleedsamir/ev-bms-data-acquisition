# This file is still under development
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd


class H5DatasetHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    def h5_print(self):
        """Recursively print all groups and datasets in the HDF5 file."""
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} | shape: {obj.shape}")
        with h5py.File(self.file_path, 'r') as f:
            f.visititems(print_attrs)
    
    def h5_to_dataframe(self, datasets_to_extract=None):
        """
        Extract specified datasets from HDF5 file and return a unified DataFrame.

        Args:
            file_path (str): path to HDF5 file
            datasets_to_extract (dict): mapping group_name -> list of dataset names to extract
                                        e.g., {'bms': ['voltage', 'current'], 'env': ['temp']}
                                        If None, extract all datasets under each group.

        Returns:
            pd.DataFrame: concatenated dataframe with columns named as group/dataset
        """
        raw_data_list = []

        with h5py.File(self.file_path, 'r') as f:
            for run_name in f.keys():
                run_group = f[run_name]
                run_data = {'run_name': run_name}

                # Iterate over requested groups
                for group_name, datasets in (datasets_to_extract or {}).items():
                    if group_name not in run_group:
                        continue
                    group = run_group[group_name]
                    # If no dataset list provided, take all datasets in the group
                    datasets = datasets or list(group.keys())
                    for ds_name in datasets:
                        if ds_name in group:
                            value = group[ds_name][:]
                            # Flatten 1D or 2D arrays as needed
                            if isinstance(value, np.ndarray) and value.ndim == 1:
                                run_data[f"{group_name}/{ds_name}"] = value
                            elif isinstance(value, np.ndarray) and value.ndim == 2:
                                # Flatten second axis as separate columns
                                for i in range(value.shape[1]):
                                    run_data[f"{group_name}/{ds_name}_{i+1}"] = value[:, i]
                            else:
                                run_data[f"{group_name}/{ds_name}"] = value
                # Include metadata if present
                if hasattr(run_group, "attrs") and run_group.attrs:
                    for k, v in run_group.attrs.items():
                        run_data[f"metadata/{k}"] = v

                raw_data_list.append(run_data)

        # Convert raw_data_list (list of dicts of arrays) to a unified DataFrame
        df_list = []
        for run in raw_data_list:
            # Ensure all arrays have the same length
            length = max([len(v) for k, v in run.items() if isinstance(v, np.ndarray)])
            df_dict = {}
            for k, v in run.items():
                if isinstance(v, np.ndarray):
                    df_dict[k] = v
                else:
                    # Broadcast scalar metadata
                    df_dict[k] = [v]*length
            df_list.append(pd.DataFrame(df_dict))

        df = pd.concat(df_list, ignore_index=True)
        return df


class DatasetManager:
    def __init__(self, X, y, features=None, target=None, test_size=0.2, random_state=42):

        self.X = X
        self.y = y

        self.features = features
        self.target = target

        self.test_size = test_size
        self.random_state = random_state

        self.scaler_X = None
        self.scaler_y = None

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    # Split Data
    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    # Apply Scaling
    def apply_scaling(self, scaler=StandardScaler()):
        if self.X_train is None:
            raise ValueError("Call split_data() before scaling.")

        self.scaler_X = scaler
        self.scaler_y = scaler.__class__()  # separate scaler instance

        # Fit on training only
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_val = self.scaler_X.transform(self.X_val)

        self.y_train = self.scaler_y.fit_transform(self.y_train)
        self.y_val = self.scaler_y.transform(self.y_val)

        return self.X_train, self.X_val, self.y_train, self.y_val

    # Save Scalers
    def save_scaler(self, path):
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not fitted yet.")

        joblib.dump({
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y
        }, path)

    # Load Scalers
    def load_scaler(self, path):
        scalers = joblib.load(path)
        self.scaler_X = scalers["scaler_X"]
        self.scaler_y = scalers["scaler_y"]

    # Transform input sample
    def transform_input(self, X):
        if self.scaler_X is None:
            raise ValueError("Scaler not loaded.")
        return self.scaler_X.transform(X)

    # Inverse transform output
    def inverse_transform_output(self, y):
        if self.scaler_y is None:
            raise ValueError("Scaler not loaded.")
        return self.scaler_y.inverse_transform(y)
