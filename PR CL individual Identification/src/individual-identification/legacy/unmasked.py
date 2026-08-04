import os
import pickle
import numpy as np
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure matplotlib backend before importing pyplot
import matplotlib
if os.getenv('DISPLAY'):
    try:
        matplotlib.use('TkAgg')
        print("Using TkAgg backend")
    except Exception:
        matplotlib.use('Agg')
        print("Using Agg backend")
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import kagglehub

from ..evaluation import get_acc, visualize_cell_frequencies

def download_czechlynx():
    path = kagglehub.dataset_download('picekl/czechlynx', force_download = False)
    return path

def main():
    # Show full lines
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    path = download_czechlynx()
    print(f"Dataset loaded to '{path}' ...")

    metadata = pd.read_csv(os.path.join(path, "CzechLynxDataset-Metadata-Real.csv"))

    transform_display = T.Compose([
        T.Resize([224, 224]),
    ])

    transform = T.Compose([
        *transform_display.transforms,
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = WildlifeDataset(path, metadata, transform=transform_display, load_label=True, col_label="unique_name")
    print(f"Loaded dataset with {len(dataset)} samples")

    # Plot a sample of the dataset
    fig = dataset.plot_grid()
    fig.set_size_inches(12, 6)
    plt.title("Czech Lynx Dataset - Sample", fontsize=20)
    plt.show()

    # Plot a specific individual
    individual_name = "lynx_202"
    idx = dataset.metadata['unique_name'] == individual_name
    idx[idx.isnull()] = False
    fig = dataset.plot_grid(idx=idx)
    fig.set_size_inches(12, 6)
    plt.title(f"Czech Lynx Dataset - Specific Individual {individual_name}", fontsize=20)
    plt.show()

    # Plot the cell frequencies
    visualize_cell_frequencies(dataset.metadata)

    # Loading the dataset
    dataset = WildlifeDataset(path, metadata, transform=transform, load_label=True, col_label="unique_name")
    dataset_database = dataset.get_subset(dataset.metadata['split-time_closed'] == 'train')
    dataset_query = dataset.get_subset(dataset.metadata['split-time_closed'] == 'test')
    n_query = len(dataset_query)

    print(f"Loaded {len(dataset_database)} database samples and {len(dataset_query)} query samples")

    # Loading the model
    name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cache paths for features
    cache_dir = os.path.join(path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    features_database_path = os.path.join(cache_dir, 'features_database.pkl')
    features_query_path = os.path.join(cache_dir, 'features_query.pkl')
    
    caching = True
    print(f"Caching of features enabled: {caching}")

    # Load cached features if they exist, otherwise compute and save them
    if caching and os.path.exists(features_database_path) and os.path.exists(features_query_path):
        print("Loading cached features...")
        with open(features_database_path, 'rb') as f:
            features_database = pickle.load(f)
        with open(features_query_path, 'rb') as f:
            features_query = pickle.load(f)
        print(f"Loaded cached features: database length {len(features_database)}, query length {len(features_query)}")
    else:
        print("Computing features (this may take a while)...")
        model = timm.create_model(name, num_classes=0, pretrained=True)
        extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=4)
        features_database = extractor(dataset_database)
        features_query = extractor(dataset_query)
        
        # Save FeatureDataset objects using pickle
        print(f"Saving features to cache: database length {len(features_database)}, query length {len(features_query)}")
        with open(features_database_path, 'wb') as f:
            pickle.dump(features_database, f)
        with open(features_query_path, 'wb') as f:
            pickle.dump(features_query, f)
        print("Features cached successfully!")

    similarity = CosineSimilarity()(features_query, features_database)

    pred_idx = similarity.argsort(axis=1)[:,-1]
    pred_scores = similarity[range(n_query), pred_idx]
    print(f"Similarity matrix: {similarity.shape}")
    print(f"Top predicted (index, score): {list(zip(pred_idx, np.round(pred_scores,3)))[:10]}")

    num_examples = 8
    top_k = 3

    dataset = WildlifeDataset(path, metadata, transform=transform_display, load_label=True, col_label="unique_name")
    dataset_database = dataset.get_subset(dataset.metadata['split-time_closed'] == 'train')
    dataset_query = dataset.get_subset(dataset.metadata['split-time_closed'] == 'test')

    print("Accuracy:")
    for k in range(1, top_k+1):
        acc = get_acc(dataset_query, dataset_database, similarity, top_k=k)
        print(f"  Top {k}: {acc:.3f}")

    pred_idx = similarity.argsort(axis=1)
    for query_idx in np.random.choice(np.arange(len(dataset_query)), num_examples, False):
        database_idx = pred_idx[query_idx, -top_k:][::-1]
        score = similarity[query_idx, database_idx]
        fig, ax = plt.subplots(1, 1+top_k, figsize=((1+top_k)*3, 3))
        
        query_data = dataset_query[query_idx]
        ax[0].imshow(query_data[0])
        ax[0].set_title(query_data[1])

        for i, idx in enumerate(database_idx):
            database_data = dataset_database[idx]
            ax[i+1].imshow(database_data[0])
            ax[i+1].set_title(database_data[1] + f": {score[i]:.3f}")

        fig.suptitle(f"Top-{top_k} Retrieval Results - Query: {query_data[1]}", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    main()