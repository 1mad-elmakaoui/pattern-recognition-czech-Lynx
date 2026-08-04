"""Cached features dataset for efficient training on pre-extracted backbone features."""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedFeaturesDataset(Dataset):
    """Dataset that loads pre-extracted backbone features from disk using memory-mapping."""
    
    def __init__(self, features_path, labels_path=None):
        """
        Args:
            features_path: Path to .npy file with features (memmapped)
            labels_path: Path to .npy file with labels
        """
        # Memory-map the features file - zero RAM usage, OS handles paging
        self.features = np.load(features_path, mmap_mode='r')
        
        # Labels are small enough to load into RAM
        # If labels_path is None, we look for a .npy file with the same name
        if labels_path is None:
            labels_path = features_path.replace('.features.npy', '.labels.npy')
            if labels_path == features_path: # Path didn't contain the suffix
                 labels_path = features_path + '.labels' # Fallback
        
        labels_raw = np.load(labels_path, allow_pickle=True)
        self.labels = np.array([label.decode('utf-8') if isinstance(label, (bytes, np.bytes_)) else label for label in labels_raw])
        
        print(f"Loaded feature dataset with {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Labels loaded: {len(self.labels)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Load single sample on-demand via memmap
        # This is extremely fast as it's just a disk offset read
        features = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]
        
        # Decode bytes to string if needed
        if isinstance(label, (bytes, np.bytes_)):
            label = label.decode('utf-8')
        
        return features, label


def extract_and_cache_features(
    backbone_model,
    dataset,
    features_cache_path,
    device,
    batch_size=128,
):
    """
    Extract backbone features for entire dataset and save to uncompressed NumPy files.
    
    Args:
        backbone_model: Model with forward_features() method
        dataset: Dataset to extract features from
        features_cache_path: Path for features .npy file (should end in .features.npy)
        device: Device to run model on
        batch_size: Batch size for extraction
    """
    from torch.utils.data import DataLoader
    
    backbone_model.eval()
    backbone_model.to(device)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for reproducibility
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Extracting features for {len(dataset)} samples...")
    
    # Get feature dimensions from first batch
    print("Determining feature dimensions from first batch...")
    first_batch = next(iter(dataloader))
    with torch.no_grad():
        sample_features = backbone_model.forward_features(first_batch[0][:1].to(device))
    
    _, num_tokens, feature_dim = sample_features.shape
    print(f"Feature shape per image: ({num_tokens}, {feature_dim})")
    
    del sample_features
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Define paths
    features_path = features_cache_path
    if not features_path.endswith('.features.npy'):
        features_path = features_path.replace('.h5', '').replace('.npy', '') + '.features.npy'
    labels_path = features_path.replace('.features.npy', '.labels.npy')
    
    # Pre-allocate output array using memmap (uncompressed, contiguous on disk)
    # This doesn't use RAM, but creates a large file on disk.
    print(f"Pre-allocating memmap file at {features_path}...")
    features_memmap = np.lib.format.open_memmap(
        features_path, 
        mode='w+', 
        dtype='float32', 
        shape=(len(dataset), num_tokens, feature_dim)
    )
    
    all_labels = []
    
    # Write features incrementally
    idx = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract features
            features = backbone_model.forward_features(images)
            batch_size_actual = len(images)
            
            # Write directly to memmap (OS handles disk sync)
            features_memmap[idx:idx+batch_size_actual] = features.cpu().numpy()
            
            # Collect labels
            if isinstance(labels, (list, tuple)):
                all_labels.extend(labels)
            elif hasattr(labels, 'tolist'):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend([labels])
            
            idx += batch_size_actual
            
            # Explicit cleanup
            del features, images
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Save labels separately
    print(f"Saving labels to {labels_path}...")
    np.save(labels_path, np.array(all_labels, dtype=object))
    
    # Ensure memmap is flushed
    del features_memmap
    
    print("Feature extraction complete!")
    return features_path, labels_path
