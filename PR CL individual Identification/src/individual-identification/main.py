"""Orchestration: config, matplotlib backend, and main pipeline."""

import os
import pickle

import dotenv
import numpy as np
import pandas as pd
import torch
import timm
import wandb
from wildlife_tools.features import DeepFeatures
from wildlife_datasets.datasets import WildlifeDataset

from .data import (
    MaskableDataset,
    PKSampler,
    denormalize,
    download_czechlynx,
    get_transforms,
)
from .evaluation import (
    compute_similarity_matrix,
    get_acc,
    plot_original_masked_augmented,
    plot_retrieval_results,
    visualize_cell_frequencies,
)
from .loss import triplet_loss
from .model import (
    prepare_model_for_finetuning_unfreeze_percentage,
    prepare_model_for_finetuning_with_efficient_probing,
    prepare_model_for_finetuning_with_linear_head,
)
from .training import train_model

# Wandb configuration (enforced when main runs, not at import)
dotenv.load_dotenv()
WANDB_PROJECT_NAME = "uj-pr-individual-identification-project"


# Configure matplotlib backend before importing pyplot
def _setup_matplotlib():
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


_setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def main():
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    if not WANDB_API_KEY:
        raise ValueError("WANDB_API_KEY is not set")

    # Show full lines
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    path = download_czechlynx()
    print(f"Dataset loaded to '{path}' ...")

    metadata = pd.read_csv(os.path.join(path, "CzechLynxDataset-Metadata-Real.csv"))

    img_size = 384
    # Display-only transform (resizing)
    transform_display = get_transforms(train=False, img_size=img_size)
    # Training transform (augmentations)
    transform_train = get_transforms(train=True, img_size=img_size)
    # Query/Database transform (inference)
    transform_infer = get_transforms(train=False, img_size=img_size)

    dataset = WildlifeDataset(path, metadata, transform=transform_display, load_label=True, col_label="unique_name")

    print(f"Loaded dataset with {len(dataset)} samples")

    no_mask = dataset.df['mask'].isnull().sum()
    percentage_no_mask = no_mask / len(dataset) * 100
    print(f"Percentage of samples without mask: {percentage_no_mask:.2f}%")

    enable_intro_visualization = False
    print(f"Intro visualization is {'enabled' if enable_intro_visualization else 'disabled'}")

    if enable_intro_visualization:
        fig = dataset.plot_grid()
        fig.set_size_inches(12, 6)
        plt.title("Czech Lynx Dataset - Sample", fontsize=20)
        plt.show()

        individual_name = "lynx_202"
        idx = dataset.metadata['unique_name'] == individual_name
        idx[idx.isnull()] = False
        fig = dataset.plot_grid(idx=idx)
        fig.set_size_inches(12, 6)
        plt.title(f"Czech Lynx Dataset - Specific Individual {individual_name}", fontsize=20)
        plt.show()

        visualize_cell_frequencies(dataset.metadata)

    # MegaDescriptor backbone
    # model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'

    # LynxV4 backbone for MegaDescriptor
    # model_name = 'hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256'

    # Dinov3 backbone
    model_name = 'vit_large_patch16_dinov3'

    # General config (shared between evaluation and finetuning)
    config = {
        # Model
        "model": model_name,

        # General
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "num_workers": 8,
        "segmentation_masks": False,
        "caching": False,
        "enable_finetuning": False,

        "preparation": "efficient_probing",

        # Required for 'efficient_probing' preparation
        "num_queries": 32,

        # Required for 'unfreeze_percentage' preparation
        # "target_trainable_percentage": 10,
    }

    # Finetuning config
    finetuning_config = {
        "num_epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "enable_gradient_clipping": True,
        "p_classes": 16,
        "k_instances": 8,
        "loss_type": "contrastive",
        "temperature": 0.07,
        "img_size": img_size
    }

    dataset_train = MaskableDataset(path, metadata, transform=transform_train, load_label=True, col_label="unique_name", enable_mask=config['segmentation_masks'])
    dataset_train_subset = dataset_train.get_subset(dataset_train.metadata['split-time_closed'] == 'train')

    dataset_plain = MaskableDataset(path, metadata, transform=transform_infer, load_label=True, col_label="unique_name", enable_mask=config['segmentation_masks'])
    dataset_database = dataset_plain.get_subset(dataset_plain.metadata['split-time_closed'] == 'train')
    dataset_query = dataset_plain.get_subset(dataset_plain.metadata['split-time_closed'] == 'test')

    n_query = len(dataset_query)
    print(f"Loaded {len(dataset_database)} database samples and {len(dataset_query)} query samples")

    cache_dir = os.path.join(path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    features_database_path = os.path.join(cache_dir, 'features_database_masked_finetuned_ep.pkl')
    features_query_path = os.path.join(cache_dir, 'features_query_masked_finetuned_ep.pkl')
    finetuned_model_path = os.path.join(cache_dir, 'finetuned_model_ep.pth')

    with wandb.init(project=WANDB_PROJECT_NAME, config=config) as run:
        if config['caching']:
            print("Caching is enabled, loading cached features and fine-tuned model...")
        else:
            print("Caching is disabled, computing features and fine-tuning model...")

        if config['caching'] and os.path.exists(features_database_path) and os.path.exists(features_query_path):
            print(f"Loading cached features from '{features_database_path}' and '{features_query_path}'...")
            with open(features_database_path, 'rb') as f:
                features_database = pickle.load(f)
            with open(features_query_path, 'rb') as f:
                features_query = pickle.load(f)
            print(f"Loaded cached features: database length {len(features_database)}, query length {len(features_query)}")
        else:
            print("Loading base model...")
            model = timm.create_model(config['model'], num_classes=0, pretrained=True)

            if config['enable_finetuning']:
                print("Preparing model for fine-tuning...")

                # Filter by split
                dataset_db = dataset.get_subset(dataset.metadata['split-time_closed'] == 'train')
                dataset_query = dataset.get_subset(dataset.metadata['split-time_closed'] == 'test')

                # Create MaskableDataset for training
                dataset_train = MaskableDataset(path, dataset_db.df, transform=transform_train, load_label=True, col_label="unique_name", enable_mask=config['segmentation_masks'])
                unique_labels = dataset_train.df[dataset_train.col_label].unique()
                print(f"Found {len(unique_labels)} unique individuals in training set")
                
                # Subset if needed (though already done for train usually)
                dataset_train_subset = dataset_train

                # Define cache path for backbone features (Memory-mapped NumPy format)
                os.makedirs("./cache", exist_ok=True)
                features_cache_path = os.path.join("./cache", f'train_features_{img_size}.features.npy')
                
                # Check if we should use cached features mode
                use_cached_training = True  # Enable feature caching by default
                
                if use_cached_training and not os.path.exists(features_cache_path):
                    # Extract features once
                    print("=" * 60)
                    print("EXTRACTING BACKBONE FEATURES")
                    print("=" * 60)
                    print(f"This is a one-time cost. Features will be cached at:")
                    print(f"  {features_cache_path}")
                    
                    # Import extraction function
                    from .cached_features import extract_and_cache_features
                    
                    # Extract features using only the backbone
                    extract_and_cache_features(
                        backbone_model=model,
                        dataset=dataset_train_subset,
                        features_cache_path=features_cache_path,
                        device=config['device'],
                        batch_size=64  # Can use larger batch for inference
                    )
                    print("Feature extraction complete!")
                    print("=" * 60)
                
                if use_cached_training:
                    # Train on cached features
                    print("Training on cached features (fast mode)...")
                    
                    from .cached_features import CachedFeaturesDataset
                    from .model import EPWithClassifier
                    
                    # Load cached features (labels are in the same HDF5 file)
                    cached_dataset = CachedFeaturesDataset(features_cache_path)
                    
                    # Get feature dimension from cached data
                    sample_features = cached_dataset[0][0]
                    feature_dim = sample_features.shape[-1]  # Last dim is feature dimension
                    
                    print(f"Cached features shape: {sample_features.shape}")
                    print(f"Feature dimension: {feature_dim}")
                    
                    # Create lightweight model (EP + Classifier only)
                    model = EPWithClassifier(
                        in_dim=feature_dim,
                        num_queries=config.get('num_queries', 32),
                        out_dim=None,  # Keep same dimension
                        num_classes=len(unique_labels)
                    )
                    
                    # Move to device
                    model = model.to(config['device'])
                    
                    print(f"Model architecture: {type(model).__name__}")
                    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                    print(f"Trainable modules: {[n for n, p in model.named_parameters() if p.requires_grad]}")
                    
                else:
                    # Traditional full-model training (slower)
                    print("Preparing full model for training...")
                    if config['preparation'] == "none":
                        model = model
                    elif config['preparation'] == "linear_head":
                        model = prepare_model_for_finetuning_with_linear_head(model)
                    elif config['preparation'] == "unfreeze_percentage":
                        model = prepare_model_for_finetuning_unfreeze_percentage(model, trainable_percentage=config['target_trainable_percentage'])
                    elif config['preparation'] == "efficient_probing":
                        model = prepare_model_for_finetuning_with_efficient_probing(
                            model,
                            num_queries=config.get('num_queries', 32),
                            num_classes=len(unique_labels)
                        )
                    else:
                        raise ValueError(f"Invalid preparation: {config['preparation']}")

                if config['caching'] and os.path.exists(finetuned_model_path):
                    print(f"Loading fine-tuned model from '{finetuned_model_path}'...")
                    model.load_state_dict(torch.load(finetuned_model_path, map_location=config['device']))
                    print("Fine-tuned model loaded successfully!")
                else:
                    print("Fine-tuning model on training dataset...")
                    config = config | finetuning_config
                    run.config.update(config)

                    try:
                        # Choose dataset and mode based on caching
                        if use_cached_training:
                            train_dataset_for_loop = cached_dataset
                            use_cached_features_flag = True
                        else:
                            train_dataset_for_loop = dataset_train_subset
                            use_cached_features_flag = False
                        
                        model = train_model(
                            model,
                            train_dataset_for_loop,
                            run=run,
                            config=config,
                            use_cached_features=use_cached_features_flag,
                        )
                    except KeyboardInterrupt:
                        print("Fine-tuning interrupted by user")
                        model.eval()

                    print(f"Saving fine-tuned model to '{finetuned_model_path}'...")
                    torch.save(model.state_dict(), finetuned_model_path)
                    print("Fine-tuned model saved successfully!")

                    artifact = wandb.Artifact(name="finetuned-model", type="model")
                    artifact.add_file(os.path.abspath(finetuned_model_path))
                    run.log_artifact(artifact)
                    print("Finetuned model logged to Wandb")
                
                # If we trained with cached features, we need to reconstruct the full model for evaluation
                # The cached training only has EP+Classifier, but evaluation needs Backbone+EP+Classifier
                if use_cached_training:
                    print("Reconstructing full model for evaluation...")
                    
                    # Load base backbone again
                    backbone_model = timm.create_model(config['model'], num_classes=0, pretrained=True)
                    
                    # Create full model with EP
                    full_model = prepare_model_for_finetuning_with_efficient_probing(
                        backbone_model,
                        num_queries=config['num_queries'],
                        num_classes=len(unique_labels)
                    )
                    
                    # Load the cached-trained EP+classifier weights into the full model
                    full_model_dict = full_model.state_dict()
                    cached_model_dict = model.state_dict()
                    
                    # Update the full model with the cached model weights
                    full_model_dict.update(cached_model_dict)
                    full_model.load_state_dict(full_model_dict)
                    
                    model = full_model
                    print("Full model reconstructed successfully!")

            print("Computing features (this may take a while)...")
            
            # Wrapper to handle tuple output (features, logits) from model
            class DeepFeaturesTupleWrapper(DeepFeatures):
                def __call__(self, dataset):
                    self.model.eval()
                    return super().__call__(dataset)

                def forward_batch(self, images):
                    output = self.model(images)
                    if isinstance(output, tuple):
                        return output[0]
                    return output

            # Monkey patch the method if we can't extend elegantly or just use a custom loop? 
            # DeepFeatures in wildlife_tools likely uses a loop.
            # Let's try to wrap the model instead.

            class TupleModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    out = self.model(x)
                    if isinstance(out, tuple):
                        return out[0]
                    return out
            
            model_wrapper = TupleModelWrapper(model)
            extractor = DeepFeatures(model_wrapper, device=config['device'], batch_size=32, num_workers=config['num_workers'])
            features_database = extractor(dataset_database)
            features_query = extractor(dataset_query)

            print(f"Saving features to cache '{features_database_path}' and '{features_query_path}': database length {len(features_database)}, query length {len(features_query)}")
            with open(features_database_path, 'wb') as f:
                pickle.dump(features_database, f)
            with open(features_query_path, 'wb') as f:
                pickle.dump(features_query, f)
            print("Features cached successfully!")

        print("Computing Euclidean distance similarity...")
        similarity = compute_similarity_matrix(
            features_query.features,
            features_database.features,
            metric="euclidean",
        )
        pred_idx = similarity.argsort(axis=1)[:, -1]
        pred_scores = similarity[range(n_query), pred_idx]
        print(f"Similarity matrix: {similarity.shape}")
        print(f"Top predicted (index, score): {list(zip(pred_idx.tolist(), [round(s, 3) for s in pred_scores.tolist()]))[:10]}")

        num_examples = 8
        top_k = 3

        # Use plain MaskableDataset for display (no mask composition) so images match labels
        dataset_display = MaskableDataset(path, metadata, transform=transform_display, load_label=True, col_label="unique_name", enable_mask=config['segmentation_masks'])
        dataset_database_display = dataset_display.get_subset(dataset_display.metadata['split-time_closed'] == 'train')
        dataset_query_display = dataset_display.get_subset(dataset_display.metadata['split-time_closed'] == 'test')

        print("Accuracy:")
        for k in range(1, top_k + 1):
            acc = get_acc(dataset_query_display, dataset_database_display, similarity, top_k=k)
            print(f"  Top {k}: {acc:.3f}")
            run.log({f"top_{k}_accuracy": acc})

        plot_retrieval_results(
            dataset_query_display,
            dataset_database_display,
            similarity,
            top_k=top_k,
            num_examples=num_examples,
        )


if __name__ == "__main__":
    main()


# Re-exports for backward compatibility
__all__ = [
    "MaskableDataset",
    "PKSampler",
    "denormalize",
    "download_czechlynx",
    "get_acc",
    "get_transforms",
    "plot_original_masked_augmented",
    "train_model",
    "triplet_loss",
    "visualize_cell_frequencies",
]
