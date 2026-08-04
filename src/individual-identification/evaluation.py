"""Evaluation responsibilities: metrics, similarity, visualization."""

from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from shapely.geometry import Point

from .data import denormalize


def compute_similarity_matrix(features_query, features_database, metric="euclidean"):
    """
    Compute similarity matrix between query and database features.
    Features can be numpy arrays or tensors; returns a tensor.

    For euclidean: normalizes L2, then similarity = -distance (higher is more similar).
    """
    if isinstance(features_query, np.ndarray):
        features_query = torch.from_numpy(features_query).float()
    if isinstance(features_database, np.ndarray):
        features_database = torch.from_numpy(features_database).float()

    features_query = F.normalize(features_query, p=2, dim=1)
    features_database = F.normalize(features_database, p=2, dim=1)

    if metric == "euclidean":
        # Higher similarity = smaller distance, so use negative distance
        similarity = -torch.cdist(features_query, features_database, p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return similarity


def get_acc(dataset_query, dataset_database, similarity, top_k=3):
    """Top-k accuracy: fraction of queries where the true identity is in the top-k retrievals."""
    if hasattr(similarity, 'cpu'):
        similarity = similarity.cpu().numpy()
    similarity = np.asarray(similarity)

    acc = []
    pred_idx = similarity.argsort(axis=1)
    for query_idx in range(len(dataset_query)):
        database_idx = pred_idx[query_idx, -top_k:][::-1]

        query_label = dataset_query.df[dataset_query.col_label].iloc[query_idx]
        database_labels = [dataset_database.df[dataset_database.col_label].iloc[di] for di in database_idx]

        acc.append(query_label in database_labels)
    return np.mean(acc)


def plot_retrieval_results(dataset_query, dataset_database, similarity, top_k=3, num_examples=8, random_state=None):
    """
    Plot retrieval grid: for a sample of queries, show query image and top-k database matches with scores.
    """
    similarity_np = similarity.cpu().numpy() if hasattr(similarity, 'cpu') else np.asarray(similarity)
    pred_idx = similarity_np.argsort(axis=1)
    n_query = len(dataset_query)
    num_examples = min(num_examples, n_query)
    rng = np.random.default_rng(random_state)
    query_indices = rng.choice(np.arange(n_query), num_examples, replace=False)

    for query_idx in query_indices:
        database_idx = pred_idx[query_idx, -top_k:][::-1].copy()
        score = similarity_np[query_idx, database_idx]
        fig, ax = plt.subplots(1, 1 + top_k, figsize=((1 + top_k) * 3, 3))

        query_data = dataset_query[query_idx]
        ax[0].imshow(query_data[0].permute(1, 2, 0))
        ax[0].set_title(query_data[1])

        for i, idx in enumerate(database_idx):
            database_data = dataset_database[idx]
            ax[i + 1].imshow(database_data[0].permute(1, 2, 0))
            ax[i + 1].set_title(database_data[1] + f": {score[i]:.3f}")

        fig.suptitle(f"Top-{top_k} Retrieval Results - Query: {query_data[1]}", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_original_masked_augmented(dataset_original, dataset_masked_display, dataset_masked_train, indices):
    """
    Plot a grid of sample images: original (no mask), masked, and augmented (masked + train augmentations).
    """
    n = len(indices)
    indices = np.random.default_rng(1).choice(indices, size=n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Masked")
    axes[0, 2].set_title("Augmented")
    for row, idx in enumerate(indices):
        img_orig, label = dataset_original[idx]
        img_masked, _ = dataset_masked_display[idx]
        img_aug, _ = dataset_masked_train[idx]
        for tensor, ax in zip(
            [img_orig, img_masked, img_aug],
            [axes[row, 0], axes[row, 1], axes[row, 2]],
        ):
            disp = denormalize(tensor.clone()).permute(1, 2, 0).clamp(0, 1)
            ax.imshow(disp)
            ax.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_cell_frequencies(data, ax=None, legend=True, point_size=(3, 70)):
    """Plot cell frequencies on a map (Czech Lynx dataset)."""
    colors = {
        ("snpa",): "red",
        ('foe_bohemia',): "blue",
        ('foe_bohemia', 'snpa'): "purple",
        ('foe_carpaths',): "green"
    }

    cell_counts = dict(data.cell_code.value_counts())
    cell_lat = dict(zip(data.cell_code, data.latitude))
    cell_lon = dict(zip(data.cell_code, data.longitude))
    cell_src = defaultdict(set)
    for cell, src in zip(data.cell_code, data.source):
        cell_src[cell].add(src)
    cell_src = {cell: tuple(sorted(set(sources))) for cell, sources in cell_src.items()}
    data_rows = [[cell_lat[cell], cell_lon[cell], cell_counts[cell], cell] for cell, cont in cell_counts.items()]

    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        data_rows,
        columns=["lat", "lon", "value", "cell"],
        geometry=[Point(lon, lat) for lat, lon, _, _ in data_rows],
        crs="EPSG:4326"
    )

    # Load Natural Earth countries data directly from URL
    ne_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(ne_url)

    # Filter to Czech Republic and surrounding countries for context
    czech_region = world[world['NAME'].isin(
        [
            'Czechia',
            'Slovakia',
            'Poland',
            'Germany',
            'Austria'
        ]
    )]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    plt.title("Czech Lynx Dataset - Cell Frequencies", fontsize=20)

    # Plot the region
    czech_region.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

    # Plot cell centers
    points_gdf.plot(
        ax=ax,
        markersize=np.interp(points_gdf["value"], [points_gdf["value"].min(), points_gdf["value"].max()], point_size),
        color=[colors[cell_src[c]] for c in points_gdf['cell']],
        alpha=0.6,
    )

    # Adjust boundaries
    miny, maxy, minx, maxx = 48.55, 51.06, 12.09, 18.86
    padding_x = (maxx - minx) * 0.15
    padding_y = (maxy - miny) * 0.15

    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)
    if legend:
        legend_handles = [Line2D([0], [0], marker='o', markerfacecolor=c, markersize=10, color='w', alpha=0.6, label=", ".join(name)) for name, c in colors.items()]
        ax.legend(handles=legend_handles)

    plt.show()
