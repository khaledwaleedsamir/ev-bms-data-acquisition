from dataset.dataset_utils import delete_run
import argparse

parser = argparse.ArgumentParser(description="Delete a run from an HDF5 dataset.")
parser.add_argument("hdf5_file", help="Path to the HDF5 file")
parser.add_argument("run_name", help="Name of the run to delete")
args = parser.parse_args()

delete_run(hdf5_file=args.hdf5_file, run_name=args.run_name)