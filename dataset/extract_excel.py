import h5py
import pandas as pd
import numpy as np
import os

def read_group(group, prefix=""):
    data = {}
    for key in group.keys():
        item = group[key]
        name = prefix + key

        if isinstance(item, h5py.Dataset):
            arr = item[()]

            if arr.ndim == 1:
                data[name] = arr

            elif arr.ndim == 2:
                rows, cols = arr.shape
                for c in range(cols):
                    data[f"{name}_{c}"] = arr[:, c]

            else:
                flat = arr.reshape(arr.shape[0], -1)
                rows, cols = flat.shape
                for c in range(cols):
                    data[f"{name}_{c}"] = flat[:, c]

        elif isinstance(item, h5py.Group):
            data.update(read_group(item, name + "/"))

    return data


def h5_to_excel(h5_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        for run_name in f.keys():  # each top-level group is a run
            run_group = f[run_name]

            print(f"Processing run: {run_name}")

            data = read_group(run_group)
            df = pd.DataFrame(data)

            out_path = os.path.join(output_folder, f"{run_name}.xlsx")
            df.to_excel(out_path, index=False)

            print(f"Saved: {out_path}")


if __name__ == "__main__":
    h5_to_excel(
        r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\hoverboard_bms_dataset.h5",
        r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel"
    )