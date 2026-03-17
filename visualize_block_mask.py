import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree
from torch.nn.attention.flex_attention import flex_attention

from nfm.modeling.block_mask import create_block_quantized_knn_mask

flex_attention = torch.compile(flex_attention)


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    n_points = 2048
    block_size = 128
    k_neighbors = 8  # number of nearest neighbors per point

    # 1. Generate random 2D points in [0, 1) x [0, 1)
    points = np.random.rand(n_points, 2).astype(np.float32)

    # 2. Create KDTree and sort points by tree structure
    kdtree = KDTree(points, leafsize=block_size)
    sorted_indices = kdtree.indices
    sorted_points = points[sorted_indices]

    # 3. Create block mask from sorted points
    block_mask = create_block_quantized_knn_mask(
        kdtree=KDTree(sorted_points, leafsize=block_size),
        points=sorted_points,
        n_points_unpadded=n_points,
        k=k_neighbors,
        block_size=block_size,
    )
    q = k = v = torch.zeros((1, 1, n_points, 64)).to("mps")
    o = flex_attention(
        q, k, v, block_mask=block_mask.to("mps")
    )  # sanity check that the block mask works with flex attention
    print(o.shape)

    num_blocks = n_points // block_size

    # 4. Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Points with block structure
    ax1 = axes[0]
    ax1.scatter(sorted_points[:, 0], sorted_points[:, 1], s=20, alpha=0.6, c="blue")

    # Draw block boundaries
    for i in range(num_blocks):
        for j in range(num_blocks):
            x_start = (j % int(np.sqrt(num_blocks))) / int(np.sqrt(num_blocks))
            y_start = (i // int(np.sqrt(num_blocks))) / int(np.sqrt(num_blocks))
            rect = patches.Rectangle(
                (x_start, y_start),
                1 / int(np.sqrt(num_blocks)),
                1 / int(np.sqrt(num_blocks)),
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                alpha=0.3,
            )
            ax1.add_patch(rect)

    # Color points by block
    colors = plt.cm.tab20(np.arange(num_blocks) % 20)
    block_assignments = np.arange(n_points) // block_size
    for block_id in range(num_blocks):
        mask = block_assignments == block_id
        ax1.scatter(
            sorted_points[mask, 0],
            sorted_points[mask, 1],
            s=50,
            alpha=0.7,
            label=f"Block {block_id}",
        )

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal")
    ax1.set_title("Sorted Points with Block Structure (16x16 blocks for 256 points)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot 2: Block attention mask visualization
    ax2 = axes[1]

    # Build attention matrix from block mask
    attention_matrix = np.zeros((n_points, n_points), dtype=bool)

    kv_num_blocks = block_mask.kv_num_blocks.squeeze()  # (num_blocks,)
    kv_indices = block_mask.kv_indices.squeeze()  # (num_blocks, max_kv_blocks)

    for q_block_id in range(num_blocks):
        num_kv = kv_num_blocks[q_block_id].item()
        kv_block_ids = kv_indices[q_block_id, :num_kv].tolist()

        # Mark which points in this Q block attend to which KV blocks
        q_start = q_block_id * block_size
        q_end = (q_block_id + 1) * block_size

        for kv_block_id in kv_block_ids:
            kv_start = kv_block_id * block_size
            kv_end = (kv_block_id + 1) * block_size
            attention_matrix[q_start:q_end, kv_start:kv_end] = True

    # Visualize attention matrix as heatmap
    im = ax2.imshow(attention_matrix, cmap="Blues", aspect="auto", origin="lower")

    # Draw block boundaries
    for i in range(num_blocks + 1):
        boundary = i * block_size
        ax2.axhline(boundary - 0.5, color="red", linewidth=0.5, alpha=0.3)
        ax2.axvline(boundary - 0.5, color="red", linewidth=0.5, alpha=0.3)

    ax2.set_xlabel("Key/Value Position")
    ax2.set_ylabel("Query Position")
    ax2.set_title(
        f"Block Attention Mask (k={k_neighbors} neighbors, block_size={block_size})"
    )
    plt.colorbar(im, ax=ax2, label="Attention")

    plt.tight_layout()
    plt.savefig("block_mask_visualization.png", dpi=150, bbox_inches="tight")
    print("Visualization saved to 'block_mask_visualization.png'")
    plt.show()

    # 5. Create individual plots for each query block showing its key/value points
    kv_num_blocks = block_mask.kv_num_blocks.squeeze()  # (num_blocks,)
    kv_indices = block_mask.kv_indices.squeeze()  # (num_blocks, max_kv_blocks)

    for q_block_id in range(num_blocks):
        num_kv = kv_num_blocks[q_block_id].item()
        kv_block_ids = kv_indices[q_block_id, :num_kv].tolist()

        # Get query points
        q_start = q_block_id * block_size
        q_end = (q_block_id + 1) * block_size
        q_points = sorted_points[q_start:q_end]

        # Get key/value points
        kv_points_list = []
        for kv_block_id in kv_block_ids:
            kv_start = kv_block_id * block_size
            kv_end = (kv_block_id + 1) * block_size
            kv_points_list.append(sorted_points[kv_start:kv_end])

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot all points in light gray
        ax.scatter(
            sorted_points[:, 0],
            sorted_points[:, 1],
            s=20,
            alpha=0.2,
            c="gray",
            label="All points",
        )

        # Plot query points in red
        ax.scatter(
            q_points[:, 0],
            q_points[:, 1],
            s=100,
            alpha=0.8,
            c="red",
            marker="o",
            edgecolors="darkred",
            linewidth=2,
            label=f"Query Block {q_block_id}",
            zorder=3,
        )

        # Plot key/value points by block
        colors = plt.cm.tab10(np.arange(len(kv_points_list)) % 10)
        for idx, (kv_block_id, kv_points) in enumerate(
            zip(kv_block_ids, kv_points_list)
        ):
            ax.scatter(
                kv_points[:, 0],
                kv_points[:, 1],
                s=100,
                alpha=0.7,
                c=[colors[idx]],
                marker="s",
                edgecolors="black",
                linewidth=1.5,
                label=f"KV Block {kv_block_id}",
                zorder=2,
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Query Block {q_block_id} (red circles) with {num_kv} KV Blocks (colored squares)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(f"block_mask_query_{q_block_id}.png", dpi=150, bbox_inches="tight")
        print(
            f"Query block {q_block_id} visualization saved to 'block_mask_query_{q_block_id}.png'"
        )
        plt.close()


if __name__ == "__main__":
    main()
