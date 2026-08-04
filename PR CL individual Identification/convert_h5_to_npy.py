import h5py
import numpy as np
from tqdm import tqdm
import os

h5_path = "./cache/train_features_384.h5"
features_out = "./cache/train_features_384.features.npy"
labels_out = "./cache/train_features_384.labels.npy"

if not os.path.exists(h5_path):
    print(f"Error: {h5_path} not found.")
    exit(1)

print(f"Converting {h5_path} to NumPy memmap...")

with h5py.File(h5_path, 'r') as f:
    features_dset = f['features']
    labels_dset = f['labels']
    
    n, t, d = features_dset.shape
    print(f"Dataset shape: ({n}, {t}, {d})")
    
    # Pre-allocate memmap
    print("Pre-allocating .features.npy...")
    mmap = np.lib.format.open_memmap(features_out, mode='w+', dtype='float32', shape=(n, t, d))
    
    # Copy in chunks to keep memory usage low
    chunk_size = 128
    for i in tqdm(range(0, n, chunk_size), desc="Converting features"):
        end = min(i + chunk_size, n)
        mmap[i:end] = features_dset[i:end]
    
    print("Saving labels...")
    labels_raw = labels_dset[:]
    # Decode if they are bytes
    labels_decoded = [l.decode('utf-8') if isinstance(l, bytes) else l for l in labels_raw]
    np.save(labels_out, np.array(labels_decoded, dtype=object))

print("\nConversion complete!")
print(f"New files: \n - {features_out}\n - {labels_out}")
