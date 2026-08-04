"""Loss responsibilities: triplet loss, contrastive loss."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb

from .data import denormalize


def triplet_loss(features, labels, margin=0.3, run=None, images=None, max_viz=1):
    """
    Triplet loss with online mining for re-identification.
    Prioritizes semi-hard negatives, falls back to hard negatives.
    """
    batch_size = features.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=features.device)

    # Normalize to unit vectors
    features = F.normalize(features, p=2, dim=1)

    # Compute pairwise Euclidean distances
    distance_matrix = torch.cdist(features, features, p=2)

    # Expand labels for broadcasting
    labels = labels.view(batch_size, 1)
    mask_same = (labels == labels.T)  # (bs, bs), True if same identity

    # Positive mask: same ID, exclude self
    eye = torch.eye(batch_size, device=features.device).bool()
    positive_mask = mask_same & ~eye

    # Negative mask: different ID
    negative_mask = ~mask_same

    # For logging distributions
    if run is not None and not run.disabled:
        pos_dists = distance_matrix[positive_mask]
        neg_dists = distance_matrix[negative_mask]
        if pos_dists.numel() > 0 and neg_dists.numel() > 0:
            run.log({"dist_pos_mean": pos_dists.mean().item(),
                     "dist_neg_mean": neg_dists.mean().item()})

    # Hard positives: max d(A, P) per anchor
    # We want to pull the farthest positive closer
    dist_pos = distance_matrix.clone()
    dist_pos[~positive_mask] = -float('inf')
    hard_pos_dist, hard_pos_indices = dist_pos.max(dim=1)  # (bs,)

    # Valid anchors must have at least one positive
    valid_anchor_mask = positive_mask.any(dim=1)
    hard_pos_dist[~valid_anchor_mask] = 0.0

    # --- Mining Strategy ---
    # We want negatives that are "hard" but not *too* hard to start with (Semi-Hard),
    # defined as: d_p < d_n < d_p + margin.
    # If no semi-hard exists, we can fall back to the hardest negative (d_n < d_p),
    # or just the closest negative.

    hard_pos_dist_exp = hard_pos_dist.unsqueeze(1)  # (bs, 1)

    # Semi-hard condition: d_n > d_p AND d_n < d_p + margin

    # Find the hardest negative (min distance) first
    dist_neg = distance_matrix.clone()
    dist_neg[~negative_mask] = float('inf')

    # Identify semi-hard negatives
    is_semi_hard = (distance_matrix > hard_pos_dist_exp) & (distance_matrix < hard_pos_dist_exp + margin)
    semi_hard_mask = negative_mask & is_semi_hard

    # If an anchor has semi-hard negatives, select the hardest among them (smallest distance)
    # If not, fall back to the hardest negative overall
    has_semi_hard = semi_hard_mask.any(dim=1)  # (bs,)

    # Vectorized approach:
    # 1. Select Hardest Negative globally
    hardest_neg_dist, hardest_neg_indices = dist_neg.min(dim=1)

    # 2. Select Hardest Semi-Hard Negative
    semi_hard_dists = dist_neg.clone()
    semi_hard_dists[~semi_hard_mask] = float('inf')
    semi_hard_neg_dist, semi_hard_neg_indices = semi_hard_dists.min(dim=1)

    # 3. Combine: Use Semi-Hard if available, else use Hardest
    final_neg_dist = torch.where(has_semi_hard, semi_hard_neg_dist, hardest_neg_dist)
    final_neg_indices = torch.where(has_semi_hard, semi_hard_neg_indices, hardest_neg_indices)

    # Visualization
    if images is not None and valid_anchor_mask.any():
        viz_count = 0
        current_max_viz = max_viz
        for i in range(batch_size):
            if not valid_anchor_mask[i] or viz_count >= current_max_viz:
                continue

            # Get Triplet Indices
            p_idx = hard_pos_indices[i]
            n_idx = final_neg_indices[i]

            # Prepare Images
            def prep_img(idx):
                img = images[idx].detach().clone().cpu()
                img = denormalize(img)
                return img.permute(1, 2, 0).numpy()

            anchor_img = prep_img(i)
            pos_img = prep_img(p_idx)
            neg_img = prep_img(n_idx)

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(anchor_img)
            axes[0].set_title(f"Anchor (ID: {labels[i].item()})")
            axes[1].imshow(pos_img)
            axes[1].set_title(f"Hard Pos (dist: {hard_pos_dist[i]:.3f})")
            axes[2].imshow(neg_img)
            axes[2].set_title(f"Selected Neg (dist: {final_neg_dist[i]:.3f})\nSemi-Hard Avail: {has_semi_hard[i].item()}")

            plt.tight_layout()

            if run is not None and not run.disabled:
                run.log({f"triplet_viz_{viz_count}": wandb.Image(fig)})

            plt.close(fig)
            viz_count += 1

    # Final triplet loss
    losses = torch.clamp(hard_pos_dist - final_neg_dist + margin, min=0.0)
    valid_losses = losses[valid_anchor_mask]

    if run is not None and not run.disabled:
        run.log({"fraction_semi_hard_used": has_semi_hard.float().mean().item(),
                 "fraction_active_triplets": (valid_losses > 1e-6).float().mean().item()})

    return valid_losses.mean() if valid_losses.numel() > 0 else torch.tensor(0.0, device=features.device)


def batch_hard_triplet_loss(features, labels, margin=0.3, run=None, images=None, max_viz=1):
    """
    Batch Hard Triplet Loss.
    For each anchor, mine the hardest positive and hardest negative in the batch.
    """
    batch_size = features.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=features.device)

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    # Compute distance matrix
    distance_matrix = torch.cdist(features, features, p=2)

    # Masks
    labels = labels.view(batch_size, 1)
    mask_same = (labels == labels.T)
    mask_diff = ~mask_same
    
    # Mask out self-distance for positives
    eye = torch.eye(batch_size, device=features.device).bool()
    mask_pos = mask_same & ~eye

    # 1. Hardest Positive: max distance to any same-ID sample
    # Set invalid entries to -inf so max ignores them.
    dist_pos = distance_matrix.clone()
    dist_pos[~mask_pos] = -float('inf')
    hard_pos_dist, hard_pos_indices = dist_pos.max(dim=1)
    
    # Handle singleton classes
    valid_anchor_mask = mask_pos.any(dim=1)
    hard_pos_dist[~valid_anchor_mask] = 0.0

    # 2. Hardest Negative: min distance to any diff-ID sample
    # Set invalid entries to +inf
    dist_neg = distance_matrix.clone()
    dist_neg[~mask_diff] = float('inf')
    hard_neg_dist, hard_neg_indices = dist_neg.min(dim=1)
    
    # Valid anchors must also have negatives (usually true unless bs=1 or all same class)
    valid_anchor_mask = valid_anchor_mask & mask_diff.any(dim=1)

    # 3. Loss = clamp(d_pos - d_neg + margin, 0)
    loss = torch.clamp(hard_pos_dist - hard_neg_dist + margin, min=0.0)
    
    # Mask out invalid anchors
    loss = loss[valid_anchor_mask]
    
    # Stats
    if run is not None and not run.disabled:
        if valid_anchor_mask.any():
            run.log({
                "hard_pos_dist_mean": hard_pos_dist[valid_anchor_mask].mean().item(),
                "hard_neg_dist_mean": hard_neg_dist[valid_anchor_mask].mean().item(),
                "fraction_active_triplets": (loss > 1e-6).float().mean().item()
            })

    # Visualization
    if images is not None and run is not None and not run.disabled and valid_anchor_mask.any() and max_viz >= 1:
        if loss.numel() > 0:
            idx_in_valid = loss.argmax()
            # Map back to original index
            # valid_anchor_mask is boolean (B,), indices are where True
            valid_indices = torch.nonzero(valid_anchor_mask, as_tuple=True)[0]
            best_i = valid_indices[idx_in_valid]
            
            p_idx = hard_pos_indices[best_i]
            n_idx = hard_neg_indices[best_i]
            
            def prep_img(idx):
                img = images[idx].detach().clone().cpu()
                img = denormalize(img)
                return img.permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(prep_img(best_i))
            axes[0].set_title(f"Anchor (ID: {labels[best_i].item()})")
            axes[1].imshow(prep_img(p_idx))
            axes[1].set_title(f"Hardest Pos (dist: {hard_pos_dist[best_i]:.3f})")
            axes[2].imshow(prep_img(n_idx))
            axes[2].set_title(f"Hardest Neg (dist: {hard_neg_dist[best_i]:.3f})\nLoss: {loss[idx_in_valid]:.3f}")
            plt.tight_layout()
            run.log({"batch_hard_viz": wandb.Image(fig)})
            plt.close(fig)

    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)



def triplet_loss_all_triplets(features, labels, margin=0.3, run=None, images=None, max_viz=1):
    """
    Triplet loss with all triplets.
    For each anchor with at least one positive and one negative, forms every
    (anchor, positive, negative) triplet and averages the hinge losses.
    """
    batch_size = features.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=features.device)

    # Normalize to unit vectors
    features = F.normalize(features, p=2, dim=1)

    # Compute pairwise Euclidean distances
    distance_matrix = torch.cdist(features, features, p=2)

    # Expand labels for broadcasting
    labels = labels.view(batch_size, 1)
    mask_same = (labels == labels.T)  # (bs, bs), True if same identity

    # Positive mask: same ID, exclude self
    eye = torch.eye(batch_size, device=features.device).bool()
    positive_mask = mask_same & ~eye

    # Negative mask: different ID
    negative_mask = ~mask_same

    # Valid anchors: at least one positive and one negative
    valid_anchor_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)

    total_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    total_count = 0
    all_losses_list = []
    best_loss_so_far = -1.0
    best_i, best_p_idx, best_n_idx = None, None, None

    for i in range(batch_size):
        if not valid_anchor_mask[i]:
            continue
        pos_indices = positive_mask[i].nonzero().squeeze(-1)
        neg_indices = negative_mask[i].nonzero().squeeze(-1)
        if pos_indices.dim() == 0:
            pos_indices = pos_indices.unsqueeze(0)
        if neg_indices.dim() == 0:
            neg_indices = neg_indices.unsqueeze(0)

        d_ap = distance_matrix[i, pos_indices]   # (P_i,)
        d_an = distance_matrix[i, neg_indices]   # (N_i,)
        losses_i = (d_ap.unsqueeze(1) - d_an.unsqueeze(0) + margin).clamp(min=0.0)  # (P_i, N_i)

        total_loss = total_loss + losses_i.sum()
        total_count += losses_i.numel()
        if losses_i.numel() > 0:
            all_losses_list.append(losses_i)
            max_i = losses_i.max().item()
            if max_i > best_loss_so_far:
                best_loss_so_far = max_i
                flat_idx = losses_i.argmax().item()
                n_neg = neg_indices.numel()
                r, c = flat_idx // n_neg, flat_idx % n_neg
                best_i = i
                best_p_idx = pos_indices[r].item()
                best_n_idx = neg_indices[c].item()

    # Visualization: triplet with largest loss
    if images is not None and run is not None and not run.disabled and best_i is not None and best_loss_so_far > 1e-6 and max_viz >= 1:
        def prep_img(idx):
            img = images[idx].detach().clone().cpu()
            img = denormalize(img)
            return img.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(prep_img(best_i))
        axes[0].set_title(f"Anchor (ID: {labels[best_i].item()})")
        axes[1].imshow(prep_img(best_p_idx))
        d_ap_best = distance_matrix[best_i, best_p_idx].item()
        axes[1].set_title(f"Pos (dist: {d_ap_best:.3f})")
        axes[2].imshow(prep_img(best_n_idx))
        d_an_best = distance_matrix[best_i, best_n_idx].item()
        axes[2].set_title(f"Neg (dist: {d_an_best:.3f})\nloss: {best_loss_so_far:.3f}")
        plt.tight_layout()
        run.log({"triplet_viz_0": wandb.Image(fig)})
        plt.close(fig)

    if total_count == 0:
        return torch.tensor(0.0, device=features.device)

    result = total_loss / total_count

    if run is not None and not run.disabled and all_losses_list:
        all_losses = torch.cat([x.flatten() for x in all_losses_list])
        fraction_active = (all_losses > 1e-6).float().mean().item()
        run.log({
            "num_triplets": total_count,
            "fraction_active_triplets": fraction_active,
        })

    return result


def contrastive_loss(features, labels, temperature=0.07, run: wandb.Run = None):
    """
    Contrastive loss (InfoNCE) for individual identification.

    Args:
        features: Tensor of shape (batch_size, feature_dim)
        labels: Tensor of shape (batch_size,) with individual IDs
        temperature: Temperature parameter for scaling

    Returns:
        Loss value
    """
    batch_size = features.shape[0]

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create mask for positive pairs (same individual)
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).float()

    # Remove diagonal (self-similarity)
    mask = mask - torch.eye(batch_size, device=mask.device)

    # For InfoNCE loss: for each anchor, we want to maximize similarity to positives
    # and minimize similarity to negatives
    exp_sim = torch.exp(similarity_matrix)

    # For each sample, sum of exp similarities for all other samples (negatives + positives)
    all_sum = torch.sum(exp_sim, dim=1, keepdim=True) - torch.exp(torch.diag(similarity_matrix)).unsqueeze(1)

    # For each positive pair, compute the loss
    # Loss = -log(exp(pos_sim) / sum(exp(all_sim)))
    pos_mask = mask > 0

    if pos_mask.sum() > 0:
        # Get positive similarities
        pos_sim = similarity_matrix * pos_mask.float()

        # For each anchor, compute loss over its positives
        losses = []
        total_num_positive_pairs = 0
        for i in range(batch_size):
            pos_indices = pos_mask[i]
            if pos_indices.sum() > 0:
                # Sum of exp similarities for positives
                pos_exp_sum = torch.sum(exp_sim[i, pos_indices])
                # Total sum (positives + negatives)
                total_exp_sum = all_sum[i]
                # Loss: -log(pos_exp / total_exp)
                loss_i = -torch.log(pos_exp_sum / (total_exp_sum + 1e-8) + 1e-8)
                losses.append(loss_i)

            total_num_positive_pairs += pos_mask[i].sum()

        average_num_positive_pairs = total_num_positive_pairs / batch_size
        if run:
            run.log({f"average_num_positive_pairs_per_sample": average_num_positive_pairs})

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=features.device, requires_grad=True)
    else:
        # No positive pairs in batch, return zero loss
        loss = torch.tensor(0.0, device=features.device, requires_grad=True)

    return loss
