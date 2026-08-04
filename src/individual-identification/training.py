"""Training responsibilities: train_model loop."""

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PKSampler
from .loss import triplet_loss, triplet_loss_all_triplets, batch_hard_triplet_loss, contrastive_loss


def train_model(
    model,
    train_dataset,
    run: wandb.Run,
    config: dict,
    use_cached_features=False,
):
    """
    Fine-tune the model using contrastive learning.

    Args:
        model: The model to fine-tune
        train_dataset: Training dataset (must have .df and .col_label unless use_cached_features=True)
        run: Wandb run
        config: Configuration for the run (device, num_epochs, batch_size, learning_rate,
                weight_decay, enable_gradient_clipping, p_classes, k_instances)
        use_cached_features: If True, train_dataset contains pre-extracted features,
                           skip image loading/processing

    Returns:
        Fine-tuned model
    """
    model = model.to(config['device'])
    model.train()

    # Create label mapping from string labels to numeric IDs
    if not use_cached_features:
        unique_labels = train_dataset.df[train_dataset.col_label].unique()
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Found {len(unique_labels)} unique individuals in training set")
    else:
        # For cached features, labels are in the HDF5 file
        # Read them directly instead of iterating through __getitem__
        sample_label = train_dataset[0][1]
        if isinstance(sample_label, str):
            # Read all labels directly from HDF5 (much faster than iterating!)
            print("Reading labels from HDF5 file...")
            all_labels_bytes = train_dataset.labels[:]  # Read entire label array at once
            # Decode bytes
            all_labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
                         for label in all_labels_bytes]
            unique_labels = list(set(all_labels))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Found {len(unique_labels)} unique individuals in cached dataset")
        else:
            # Labels already numeric
            label_to_id = None
            print(f"Using numeric labels from cached dataset")

    # Use Adam optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    print(f"Starting fine-tuning for {config['num_epochs']} epochs...")

    # COUPLED PARAMETERS: Ensure batch_size = P * K
    p = config['p_classes']
    k = config['k_instances']
    effective_batch_size = p * k
    
    if config['batch_size'] != effective_batch_size:
        raise ValueError(f"batch_size must be equal to P * K = {effective_batch_size}")

    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        num_batches = 0

        num_workers = config.get('num_workers', 0)
        
        # Use PKSampler for both modes to ensure metric learning batch structure
        sampler = PKSampler(train_dataset, p=p, k=k, shuffle=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=sampler,
            num_workers=0 if use_cached_features else num_workers,
            pin_memory=True if config['device'] == 'cuda' else False
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        tokens, images = None, None
        for batch_idx, batch_data in enumerate(pbar):
            if not isinstance(batch_data, (list, tuple)):
                raise ValueError("Dataset must return (data, label) tuples")

            if use_cached_features:
                # Cached features: data is already tokens
                tokens, labels = batch_data
                tokens = tokens.to(config['device'])
            else:
                # Regular images: need to process through model
                images, labels = batch_data
                images = images.to(config['device'])

            # Convert string labels to numeric IDs if needed
            if label_to_id is not None:
                if isinstance(labels, (list, tuple)):
                    if isinstance(labels[0], str):
                        labels = torch.tensor([label_to_id.get(str(l), 0) for l in labels], device=config['device'])
                    else:
                        labels = torch.tensor(labels, device=config['device'])
                elif isinstance(labels, np.ndarray):
                    labels = torch.tensor([label_to_id.get(str(l), 0) for l in labels], device=config['device'])
                else:
                    labels = labels.to(config['device'])
            else:
                # Already numeric
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, device=config['device'])
                else:
                    labels = labels.to(config['device'])

            # Forward pass
            optimizer.zero_grad()
            
            if use_cached_features:
                # Model expects tokens directly
                outputs = model(tokens)
            else:
                # Model expects images
                outputs = model(images)
            if isinstance(outputs, tuple):
                features, logits = outputs
            else:
                features = outputs
                logits = None

            # Compute loss
            loss_type = config.get("loss_type", "batch_hard")
            
            # Metric Loss
            if loss_type == "contrastive":
                metric_loss = contrastive_loss(
                    features, 
                    labels, 
                    temperature=config.get("temperature", 0.07),
                    run=run
                )
            elif loss_type == "triplet_all":
                # For cached features, we don't have images for visualization
                images_for_loss = None if use_cached_features else images
                metric_loss = triplet_loss_all_triplets(features, labels, run=run, images=images_for_loss)
            elif loss_type == "triplet":
                images_for_loss = None if use_cached_features else images
                metric_loss = triplet_loss(features, labels, run=run, images=images_for_loss)
            elif loss_type == "batch_hard":
                images_for_loss = None if use_cached_features else images
                metric_loss = batch_hard_triplet_loss(features, labels, run=run, images=images_for_loss)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            # Identity Loss (Cross Entropy)
            # loss_id = torch.tensor(0.0, device=config['device'])
            # if logits is not None:
            #     loss_id = torch.nn.functional.cross_entropy(logits, labels)

            # Total Loss
            loss = metric_loss

            # Backward pass
            loss.backward()

            if config['enable_gradient_clipping']:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'metric': f'{metric_loss.item():.4f}',
                # 'id': f'{loss_id.item():.4f}'
            })

            # Track loss
            run.log({
                "loss": loss.item(),
                "loss_metric": metric_loss.item(),
                # "loss_id": loss_id.item()
            })

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Average Loss: {avg_loss:.4f}")

    model.eval()
    return model
