"""Complete physical windows with hidden geometry confined to target rasters."""

import hashlib

import lance
import numpy as np
from torch.utils.data import Dataset

from nfm.data.efd import elliptic_fourier_descriptors
from nfm.data.geometry import radial_polygons


HISTOGRAM_EDGES = ([16, 32, 64, 128, 256], [1.25, 1.5, 2, 3], [0.5, 0.65, 0.8, 0.9])
HISTOGRAM_SIZES = tuple(len(edges) + 1 for edges in HISTOGRAM_EDGES)


def block_mask(rng, grid_size, ratio, block_size, anchor_tile):
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    desired = min(grid_size**2 - 1, max(1, round(grid_size**2 * ratio)))
    for _ in range(grid_size**2 * 20):
        y, x = rng.integers(0, grid_size - block_size + 1, size=2)
        candidates = [
            (j, i)
            for j in range(y, y + block_size)
            for i in range(x, x + block_size)
            if j * grid_size + i != anchor_tile and not mask[j, i]
        ]
        for j, i in candidates[: desired - int(mask.sum())]:
            mask[j, i] = True
        if mask.sum() == desired:
            return mask.ravel()
    raise RuntimeError("Could not draw a spatial mask")


def shape_categories(polygons):
    centered = polygons - polygons.mean(1, keepdims=True)
    following = np.roll(centered, -1, axis=1)
    area = (
        np.abs(
            (
                centered[..., 0] * following[..., 1]
                - centered[..., 1] * following[..., 0]
            ).sum(1)
        )
        / 2
    )
    perimeter = np.linalg.norm(following - centered, axis=-1).sum(1)
    covariance = centered.transpose(0, 2, 1) @ centered / centered.shape[1]
    axes = np.linalg.eigvalsh(covariance).clip(1e-12)
    if not np.isfinite(polygons).all() or np.any(area <= 0) or np.any(perimeter <= 0):
        raise ValueError("Invalid contour in a regional window")
    values = (
        area,
        np.sqrt(axes[:, 1] / axes[:, 0]),
        np.clip(4 * np.pi * area / perimeter**2, 0, 1),
    )
    return [
        np.digitize(value, edges)
        for value, edges in zip(values, HISTOGRAM_EDGES, strict=True)
    ]


def make_sample(
    points,
    polygons,
    origin,
    mask,
    *,
    grid_size,
    tile_size_um,
    fine_bins,
    efd_order,
    slide_id,
):
    """No hidden cell coordinates, IDs, or shapes survive into the input payload."""
    side = grid_size * tile_size_um
    local = np.asarray(points, dtype=np.float64) - origin
    if not np.all((local >= 0) & (local < side)):
        raise ValueError("Window points lie outside its physical bounds")
    tiles = np.floor(local / tile_size_um).astype(np.int64)
    tile_ids = tiles[:, 1] * grid_size + tiles[:, 0]
    fine = np.floor((local / tile_size_um - tiles) * fine_bins).astype(np.int64)
    counts = np.zeros((grid_size**2, fine_bins, fine_bins), dtype=np.float32)
    np.add.at(counts, (tile_ids, fine[:, 1], fine[:, 0]), 1)
    morphology = np.zeros((grid_size**2, sum(HISTOGRAM_SIZES)), dtype=np.float32)
    offset = 0
    for categories, size in zip(
        shape_categories(polygons), HISTOGRAM_SIZES, strict=True
    ):
        np.add.at(morphology, (tile_ids, categories + offset), 1)
        offset += size
    visible = ~mask[tile_ids]
    if not visible.any():
        raise ValueError("The visible anchor must survive masking")
    coefficients = elliptic_fourier_descriptors(polygons[visible], efd_order)
    return {
        "efds": coefficients.reshape(int(visible.sum()), -1).astype(np.float32),
        "pos": local[visible],
        "mask": mask,
        "targets": {"counts": counts, "morphology": morphology},
        "slide_id": str(slide_id),
    }


class RegionalNucleiDataset(Dataset):
    def __init__(
        self,
        slides,
        nuclei_path: str,
        grid_size: int = 8,
        tile_size_um: float = 64,
        fine_bins: int = 8,
        efd_order: int = 16,
        mask_ratio: float = 0.4,
        mask_block_size: int = 2,
        initial_query: int = 4096,
        max_query: int = 262144,
        max_cells: int = 32768,
        deterministic: bool = False,
        seed: int = 0,
    ):
        if (
            grid_size < 2
            or not 1 <= mask_block_size <= grid_size
            or not 0 < mask_ratio < 1
        ):
            raise ValueError("Invalid window/mask dimensions")
        if (
            min(tile_size_um, fine_bins, efd_order, initial_query, max_query, max_cells)
            <= 0
        ):
            raise ValueError("Window, retrieval, and descriptor sizes must be positive")
        if initial_query > max_query or fine_bins & (fine_bins - 1):
            raise ValueError(
                "Query budget is invalid or fine_bins is not a power of two"
            )
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.grid_size, self.tile_size_um = grid_size, tile_size_um
        self.fine_bins, self.efd_order = fine_bins, efd_order
        self.mask_ratio, self.mask_block_size = mask_ratio, mask_block_size
        self.initial_query, self.max_query, self.max_cells = (
            initial_query,
            max_query,
            max_cells,
        )
        self.deterministic, self.seed = deterministic, seed

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index):
        slide = self.slides.iloc[index]
        seed = (
            int.from_bytes(
                hashlib.blake2b(
                    f"{self.seed}:{slide.id}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            if self.deterministic
            else int(np.random.randint(0, 2**32))
        )
        rng = np.random.default_rng(seed)
        store = lance.dataset(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}/nuclei.lance"
        )
        census = store.count_rows()
        if not census:
            raise ValueError(f"Empty Lance slide: {slide.id}")
        mpp = np.asarray([slide.mpp_x, slide.mpp_y], dtype=np.float64)
        if not np.isfinite(mpp).all() or np.any(mpp <= 0):
            raise ValueError(f"Invalid MPP: {slide.id}")
        anchor = (
            np.asarray(
                store.take([int(rng.integers(census))], columns=["points"])["points"][
                    0
                ].as_py()
            )
            * mpp
        )
        side = self.grid_size * self.tile_size_um
        origin = anchor - rng.uniform(0, side, size=2)
        query = ((origin + side / 2) / mpp).astype(np.float32)
        corners = origin + np.array([[0, 0], [side, 0], [0, side], [side, side]])
        covering_radius2 = (
            np.square(corners / mpp - query.astype(np.float64)).sum(1).max()
        )
        limit = min(census, self.max_query)
        count = min(limit, self.initial_query)
        while True:
            table = store.to_table(
                columns=["points"],
                with_row_id=True,
                disable_scoring_autoprojection=True,
                nearest={
                    "column": "points",
                    "q": query.astype(np.float32),
                    "k": count,
                    "metric": "euclidean",
                    "use_index": False,
                },
            )
            pixels = np.asarray(table["points"].to_pylist(), dtype=np.float64)
            radius2 = np.square(pixels - query).sum(1).max(initial=0)
            if len(pixels) == census or radius2 > covering_radius2 * (1 + 1e-6):
                break
            if count == limit:
                raise ValueError(
                    f"Uncertified physical window on {slide.id}; increase max_query"
                )
            count = min(limit, count * 2)
        physical = pixels * mpp
        inside = np.all((physical >= origin) & (physical < origin + side), axis=1)
        row_ids = np.asarray(table["_rowid"].to_pylist(), dtype=np.uint64)[inside]
        if not len(row_ids) or len(row_ids) > self.max_cells:
            raise ValueError(
                f"Physical window has {len(row_ids)} cells; check/increase max_cells"
            )
        # Fetch contours only after certifying the complete window, in persistent-row-ID order.
        take_rows = getattr(store, "take_rows", None) or store._take_rows
        radii = np.asarray(
            take_rows(row_ids.tolist(), columns=["radial_distances"])[
                "radial_distances"
            ].to_pylist(),
            dtype=np.float64,
        )
        polygons = radial_polygons(pixels[inside], radii) * mpp
        anchor_xy = np.floor((anchor - origin) / self.tile_size_um).astype(int)
        anchor_tile = int(anchor_xy[1] * self.grid_size + anchor_xy[0])
        mask = block_mask(
            rng, self.grid_size, self.mask_ratio, self.mask_block_size, anchor_tile
        )
        result = make_sample(
            physical[inside],
            polygons,
            origin,
            mask,
            grid_size=self.grid_size,
            tile_size_um=self.tile_size_um,
            fine_bins=self.fine_bins,
            efd_order=self.efd_order,
            slide_id=slide.id,
        )
        result["window_origin_um"] = origin
        return result
