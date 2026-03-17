import matplotlib.pyplot as plt
import numpy as np

# Set parameters
N = 50
np.random.seed(42)

# Create two arrays of size N with sorted random numbers
array1 = np.sort(np.random.randn(N) * 10 + 50)
array2 = np.sort(np.random.randn(N) * 10 + 50)

print(f"Array 1 (first 10): {array1[:10]}")
print(f"Array 2 (first 10): {array2[:10]}")

# Compute pairwise distances
# Shape will be (N, N) where element [i, j] is |array1[i] - array2[j]|
pairwise_distances = np.abs(array1[:, np.newaxis] - array2[np.newaxis, :])

print(f"Pairwise distance matrix shape: {pairwise_distances.shape}")

# Visualize as a matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pairwise_distances, cmap="viridis", aspect="auto")
ax.set_xlabel("Array 2 Index")
ax.set_ylabel("Array 1 Index")
ax.set_title(f"Pairwise Distance Matrix (N={N})")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Distance")

plt.tight_layout()
plt.savefig("pairwise_distance_matrix.png", dpi=150)
plt.show()

print(f"\nDistance matrix statistics:")
print(f"  Min distance: {np.min(pairwise_distances):.4f}")
print(f"  Max distance: {np.max(pairwise_distances):.4f}")
print(f"  Mean distance: {np.mean(pairwise_distances):.4f}")
