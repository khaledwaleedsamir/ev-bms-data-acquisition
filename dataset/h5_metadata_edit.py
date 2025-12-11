import h5py

h5_path = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\hoverboard_bms_dataset.h5"


with h5py.File(h5_path, "r+") as f:
    run_name = "run_003"
    run_group = f[run_name]
    # List subgroups
    print(list(run_group.keys()))
    metadata_group = run_group["metadata"]
    metadata_group.attrs["description"] = "Running hoverboard at a constant full speed from 100% SOC to 50% SOC."
    for attr in metadata_group.attrs:
        print(attr, ":", metadata_group.attrs[attr])