import h5py
import argparse
import os

parser = argparse.ArgumentParser(description="Combine multiple HDF5 files into one.")
parser.add_argument("inputs", nargs="+", help="Paths to input HDF5 files")
parser.add_argument("--output", required=True, help="Path to output HDF5 file")
args = parser.parse_args()

with h5py.File(args.output, 'w') as h5fw:
    for input_path in args.inputs:
        prefix = os.path.splitext(os.path.basename(input_path))[0]
        with h5py.File(input_path, 'r') as h5fr:
            for obj_name in h5fr.keys():
                out_name = f"{prefix}__{obj_name}"
                if out_name in h5fw:
                    raise ValueError(f"Unexpected name collision: '{out_name}'")
                h5fr.copy(obj_name, h5fw, name=out_name)

print(f"Combined {len(args.inputs)} files into {args.output}")