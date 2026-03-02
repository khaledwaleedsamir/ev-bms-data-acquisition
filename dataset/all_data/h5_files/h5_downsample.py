import h5py
import numpy as np
import argparse
from pathlib import Path

def downsample_h5(input_path, output_path, ratio=10, target_runs=None):
    """
    Downsample HDF5 file containing BMS data.
    
    Args:
        input_path: Path to input H5 file
        output_path: Path to output H5 file
        ratio: Downsampling ratio (keep every nth sample)
        target_runs: List of run names to downsample (None = all runs)
    """
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        
        for run_name in f_in.keys():
            run_group_in = f_in[run_name]
            run_group_out = f_out.create_group(run_name)
            
            # Check if this run should be downsampled
            should_downsample = (target_runs is None) or (run_name in target_runs)
            
            # Copy attributes (metadata)
            for attr_name, attr_value in run_group_in.attrs.items():
                run_group_out.attrs[attr_name] = attr_value
            
            # Get original length from timestamp
            original_length = len(run_group_in['timestamp_ms'][:])
            
            if should_downsample:
                # Create downsampling indices
                indices = np.arange(0, original_length, ratio)
                new_length = len(indices)
                print(f"Downsampling {run_name}: {original_length} -> {new_length} samples")
            else:
                indices = np.arange(original_length)
                new_length = original_length
                print(f"Keeping {run_name}: {original_length} samples (no downsampling)")
            
            # Process top-level datasets (like timestamp_ms)
            for dataset_name in run_group_in.keys():
                if isinstance(run_group_in[dataset_name], h5py.Dataset):
                    # It's a dataset at run level
                    data = run_group_in[dataset_name][:]
                    downsampled_data = data[indices]
                    run_group_out.create_dataset(dataset_name, data=downsampled_data)
                    
                elif isinstance(run_group_in[dataset_name], h5py.Group):
                    # It's a group (like 'bms')
                    subgroup_in = run_group_in[dataset_name]
                    subgroup_out = run_group_out.create_group(dataset_name)
                    
                    # Copy subgroup attributes
                    for attr_name, attr_value in subgroup_in.attrs.items():
                        subgroup_out.attrs[attr_name] = attr_value
                    
                    # Process datasets within the subgroup
                    for sub_dataset_name in subgroup_in.keys():
                        if isinstance(subgroup_in[sub_dataset_name], h5py.Dataset):
                            data = subgroup_in[sub_dataset_name][:]
                            downsampled_data = data[indices]
                            subgroup_out.create_dataset(sub_dataset_name, data=downsampled_data)
    
    print(f"\nDownsampled H5 file saved to: {output_path}")


def analyze_sampling_rate(input_path):
    """Analyze sampling rates across runs to help decide downsampling ratio."""
    
    print("=" * 60)
    print("SAMPLING RATE ANALYSIS")
    print("=" * 60)
    
    with h5py.File(input_path, 'r') as f:
        for run_name in f.keys():
            run_group = f[run_name]
            timestamps = run_group['timestamp_ms'][:]
            
            # Calculate time differences
            time_diffs = np.diff(timestamps)
            
            # Statistics
            num_samples = len(timestamps)
            duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0
            avg_sample_rate = num_samples / duration_sec if duration_sec > 0 else 0
            avg_interval_ms = np.mean(time_diffs)
            min_interval_ms = np.min(time_diffs)
            max_interval_ms = np.max(time_diffs)
            
            print(f"\n{run_name}:")
            print(f"  Samples: {num_samples}")
            print(f"  Duration: {duration_sec:.1f} sec ({duration_sec/60:.1f} min)")
            print(f"  Avg sample rate: {avg_sample_rate:.1f} Hz")
            print(f"  Avg interval: {avg_interval_ms:.1f} ms")
            print(f"  Min interval: {min_interval_ms:.1f} ms")
            print(f"  Max interval: {max_interval_ms:.1f} ms")


def verify_downsampled_file(original_path, downsampled_path):
    """Verify the downsampled file structure and data."""
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    with h5py.File(original_path, 'r') as f_orig, h5py.File(downsampled_path, 'r') as f_down:
        
        print(f"\nOriginal runs: {list(f_orig.keys())}")
        print(f"Downsampled runs: {list(f_down.keys())}")
        
        for run_name in f_down.keys():
            orig_len = len(f_orig[run_name]['timestamp_ms'][:])
            down_len = len(f_down[run_name]['timestamp_ms'][:])
            ratio = orig_len / down_len if down_len > 0 else 0
            
            print(f"\n{run_name}:")
            print(f"  Original: {orig_len} samples")
            print(f"  Downsampled: {down_len} samples")
            print(f"  Actual ratio: {ratio:.1f}x")
            
            # Verify BMS data exists
            if 'bms' in f_down[run_name]:
                bms_datasets = list(f_down[run_name]['bms'].keys())
                print(f"  BMS datasets: {bms_datasets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample an HDF5 BMS dataset file.")
    parser.add_argument("input", help="Path to input H5 file")
    parser.add_argument("output", help="Path to output H5 file")
    parser.add_argument("--ratio", type=int, default=10, help="Downsampling ratio (default: 10)")
    parser.add_argument("--runs", nargs='+', default=None, 
                        help="Specific runs to downsample (default: all runs)")
    parser.add_argument("--analyze", action="store_true", 
                        help="Analyze sampling rates before downsampling")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output file after downsampling")
    
    args = parser.parse_args()
    
    # Analyze first if requested
    if args.analyze:
        analyze_sampling_rate(args.input)
        print("\n")
    
    # Perform downsampling
    downsample_h5(
        input_path=args.input,
        output_path=args.output,
        ratio=args.ratio,
        target_runs=args.runs
    )
    
    # Verify if requested
    if args.verify:
        verify_downsampled_file(args.input, args.output)
    
    print("\nDownsampling complete!")