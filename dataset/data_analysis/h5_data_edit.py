import h5py
import os

file_path = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\hoverboard_bms_dataset.h5"


# with h5py.File(h5_path, "r+") as f:
#     run_name = "run_003"
#     run_group = f[run_name]
#     # List subgroups
#     print(list(run_group.keys()))
#     metadata_group = run_group["metadata"]
#     metadata_group.attrs["description"] = "Running hoverboard at a constant full speed from 100% SOC to 50% SOC."
#     for attr in metadata_group.attrs:
#         print(attr, ":", metadata_group.attrs[attr])

group_to_delete = '/run_004'

# Ensure the file exists for demonstration
if not os.path.exists(file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_group(group_to_delete)
        f.create_dataset(f"{group_to_delete}/data", data=[1, 2, 3])

# Open the HDF5 file in append mode and delete the group
with h5py.File(file_path, 'a') as f:
    if group_to_delete in f:
        del f[group_to_delete]
        print(f"Group '{group_to_delete}' deleted.")
    else:
        print(f"Group '{group_to_delete}' not found.")

# Verify deletion (optional)
with h5py.File(file_path, 'r') as f:
    print(f"Contents after deletion: {list(f.keys())}")
