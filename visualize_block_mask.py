import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.nn.attention.flex_attention import flex_attention

from nfm.modeling.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


flex_attention = torch.compile(flex_attention)


def simulate_and_visualize():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "Warning: flex_attention typically requires a CUDA device. Running on CPU may fail."
        )

    BLOCK_SIZE = 128
    K = 8
    sequence_lengths = [1000, 24, 240, 820]

    neighbor_indices_list = []
    all_points_list = []

    current_idx = 0

    for seq_len in sequence_lengths:
        points = np.random.rand(seq_len, 2).astype(np.float32)

        # Pass the global offset so the sorter can align splits to the 128-token grid
        sort_indices = block_spatial_sort(points, BLOCK_SIZE, global_offset=current_idx)
        points = points[sort_indices]

        nbrs = NearestNeighbors(n_neighbors=K, algorithm="auto").fit(points)
        _, knn = nbrs.kneighbors(points)

        neighbor_indices_list.append(
            torch.tensor(knn, dtype=torch.int32, device=device)
        )
        all_points_list.append(points)

        current_idx += seq_len

    block_mask = create_ragged_block_quantized_knn_mask(
        neighbor_indices_list, BLOCK_SIZE
    ).to(device)

    N_total = sum(sequence_lengths)
    D = 32

    q = torch.randn(
        1, 1, N_total, D, dtype=torch.float16, device=device, requires_grad=True
    )
    k = torch.randn(
        1, 1, N_total, D, dtype=torch.float16, device=device, requires_grad=True
    )
    v = torch.randn(
        1, 1, N_total, D, dtype=torch.float16, device=device, requires_grad=True
    )

    bias = torch.zeros(
        1, 1, N_total, N_total, dtype=torch.float16, device=device, requires_grad=True
    )

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b, h, q_idx, kv_idx]

    out = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
    loss = out.sum()
    loss.backward()

    evaluated_mask = (bias.grad != 0).squeeze().cpu().numpy()

    # ==========================================
    # 3. Visualization: Matrix + 2D Points
    # ==========================================
    num_docs = len(sequence_lengths)

    fig = plt.figure(figsize=(max(12, 4 * num_docs), 14))
    gs = fig.add_gridspec(2, num_docs, height_ratios=[1.5, 1])

    # --- Matrix ---
    ax_mat = fig.add_subplot(gs[0, :])
    ax_mat.imshow(evaluated_mask, cmap="Blues", interpolation="none")

    current_idx = 0
    for slen in sequence_lengths:
        current_idx += slen
        ax_mat.axhline(current_idx - 0.5, color="red", linewidth=2, linestyle="-")
        ax_mat.axvline(current_idx - 0.5, color="red", linewidth=2, linestyle="-")

    for i in range(0, N_total, BLOCK_SIZE):
        ax_mat.axhline(i - 0.5, color="gray", linewidth=0.5, linestyle="--")
        ax_mat.axvline(i - 0.5, color="gray", linewidth=0.5, linestyle="--")

    ax_mat.set_title(
        f"Real Flex Attention Mask (via Backprop)\nBlock Size = {BLOCK_SIZE}, K = {K}\nRed = Document Boundaries, Dashed = Blocks"
    )
    ax_mat.set_xlabel("Key Token Index")
    ax_mat.set_ylabel("Query Token Index")
    ticks = np.arange(0, N_total, BLOCK_SIZE)
    ax_mat.set_xticks(ticks)
    ax_mat.set_yticks(ticks)

    # --- 2D Spatial Points (Global Block Mapped) ---
    current_idx = 0
    cmap = plt.get_cmap("tab20")

    for doc_id, seq_len in enumerate(sequence_lengths):
        ax_2d = fig.add_subplot(gs[1, doc_id])
        pts = all_points_list[doc_id]

        # Determine exactly which global blocks this document spans
        first_global_block = current_idx // BLOCK_SIZE
        last_global_block = (current_idx + seq_len - 1) // BLOCK_SIZE
        doc_global_blocks = list(range(first_global_block, last_global_block + 1))

        centroids = {}

        # Plot points grouped by their Global Block assignment
        for gb in doc_global_blocks:
            token_start_in_doc = max(0, gb * BLOCK_SIZE - current_idx)
            token_end_in_doc = min(seq_len, (gb + 1) * BLOCK_SIZE - current_idx)
            b_pts = pts[token_start_in_doc:token_end_in_doc]

            if len(b_pts) > 0:
                ax_2d.scatter(
                    b_pts[:, 0], b_pts[:, 1], color=cmap(gb % 20), s=15, alpha=0.7
                )
                centroid = b_pts.mean(axis=0)
                centroids[gb] = centroid
                ax_2d.text(
                    centroid[0],
                    centroid[1],
                    str(gb),
                    fontsize=10,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
                )

        # Draw arrows based on the global evaluated mask
        for q_gb in doc_global_blocks:
            for k_gb in doc_global_blocks:
                if q_gb == k_gb:
                    continue

                q_start = max(current_idx, q_gb * BLOCK_SIZE)
                q_end = min(current_idx + seq_len, (q_gb + 1) * BLOCK_SIZE)
                k_start = max(current_idx, k_gb * BLOCK_SIZE)
                k_end = min(current_idx + seq_len, (k_gb + 1) * BLOCK_SIZE)

                if q_start < q_end and k_start < k_end:
                    if evaluated_mask[q_start:q_end, k_start:k_end].any():
                        if q_gb in centroids and k_gb in centroids:
                            cq, ck = centroids[q_gb], centroids[k_gb]
                            ax_2d.annotate(
                                "",
                                xy=ck,
                                xytext=cq,
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="red",
                                    alpha=0.4,
                                    linewidth=1.5,
                                    shrinkA=8,
                                    shrinkB=8,
                                ),
                            )

        ax_2d.set_title(
            f"Doc {doc_id} Spatial Keys\nGlobal Blocks: {first_global_block} - {last_global_block}"
        )
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])
        ax_2d.set_aspect("equal", adjustable="box")

        current_idx += seq_len

    plt.tight_layout()
    plt.savefig("spatial_attention_visualization.png")
    plt.show()


if __name__ == "__main__":
    simulate_and_visualize()
