"""Clustering of nucleus tokens into cells, and tracks of cells across views.

Cells start from the farthest point sampling cells of the data loader and are
refined by SLIC: each nucleus moves to the cell of its view whose center is
closest in position and in token, and the centers are recomputed. The tokens
are detached, so the encoder cannot shape the cells to ease its objective.
"""

import torch
from torch import Tensor, nn

from nfm.modeling.pooling import segment_sum


def segment_mean(values: Tensor, segments: Tensor, count: int) -> Tensor:
    """Means of the rows of values in each segment, 0 for empty ones."""
    sizes = torch.bincount(segments, minlength=count).clamp_min(1)
    return segment_sum(values, segments, count) / sizes[:, None].to(values.dtype)


def _starts(lengths: Tensor) -> Tensor:
    return lengths.cumsum(0) - lengths


class Clustering(nn.Module):
    """SLIC on the nucleus tokens, from the cells of the data loader.

    The distance of a nucleus to a cell is `(1 - mix) * spatial + mix * token`.
    Each term is the squared distance to the cell center, divided by its mean
    contrast over the nuclei of the view: the distance to the spatially nearest
    other initial cell less the distance to the own one. A typical nucleus thus
    prefers its own cell over the neighboring one by about 1 in each term,
    whatever the scale of the view or the tokens. A `mix` of 0 is spatial k-means, 0.5 weighs both alike
    and 1 uses the tokens only. Any cell of its view can take a nucleus. The tokens are randomly
    projected to `projection_dim` dimensions, which keeps their distances, for
    memory.
    """

    def __init__(self, dim: int, iterations: int = 5, projection_dim: int = 64):
        super().__init__()
        self.iterations = iterations
        generator = torch.Generator().manual_seed(0)
        projection = torch.randn(dim, projection_dim, generator=generator)
        # A Johnson-Lindenstrauss map keeps squared distances. It is fixed by its
        # seed, so it is not saved, and checkpoints without it load.
        self.register_buffer(
            "projection", projection / projection_dim**0.5, persistent=False
        )

    @torch.no_grad()
    def forward(
        self, tokens: Tensor, batch: dict[str, Tensor], mix: float
    ) -> tuple[Tensor, int, Tensor]:
        """Cells of the nuclei, numbered from 0 without empty ones.

        Returns:
            The cell of each nucleus, the number of cells and the share of the
            nuclei that left their initial cell.
        """
        initial, count = batch["cells"], int(batch["cell_count"])
        cells = initial
        if self.iterations > 0 and len(tokens) > 1:
            cells = self.refine(tokens, batch, mix)
        moved = (cells != initial).float().mean()
        used = torch.bincount(cells, minlength=count) > 0
        renumber = used.long().cumsum(0) - 1
        return renumber[cells], int(used.sum()), moved

    def refine(self, tokens: Tensor, batch: dict[str, Tensor], mix: float) -> Tensor:
        device = tokens.device
        lengths = batch["lengths"].to(device)
        cell_lengths = batch["cell_lengths"].to(device)
        cells, count = batch["cells"], int(batch["cell_count"])
        views = len(lengths)
        token_view = torch.repeat_interleave(
            torch.arange(views, device=device), lengths
        )
        token_local = torch.arange(len(tokens), device=device)
        token_local -= _starts(lengths)[token_view]
        cell_start = _starts(cell_lengths)
        cell_view = torch.repeat_interleave(
            torch.arange(views, device=device), cell_lengths
        )
        cell_local = torch.arange(count, device=device) - cell_start[cell_view]
        width = int(cell_lengths.max())
        exists = torch.arange(width, device=device) < cell_lengths[:, None]

        def padded(values: Tensor, view: Tensor, local: Tensor, rows: int) -> Tensor:
            out = values.new_zeros(views, rows, values.size(1))
            out[view, local] = values
            return out

        rows = int(lengths.max())
        positions = batch["positions"].float()
        x = tokens.detach().float()
        x = (x - x.mean(0)) @ self.projection

        def squared(values: Tensor, cells: Tensor) -> Tensor:
            """Squared distances of the nuclei to the centers of their view's cells."""
            centers = segment_mean(values, cells, count)
            return torch.cdist(
                padded(values, token_view, token_local, rows),
                padded(centers, cell_view, cell_local, width),
            ).square()

        def per_view_mean(values: Tensor) -> Tensor:
            mean = segment_mean(values[:, None], token_view, views)[:, 0]
            return mean.clamp_min(1e-12)[:, None, None]

        # The scale of each term is its mean contrast between the spatially
        # nearest other initial cell and the own one, so a token contrast as large
        # as between neighboring cells weighs as much as their spatial contrast.
        spatial = squared(positions, cells)
        own = (cells - cell_start[token_view])[:, None]
        to_cells = spatial[token_view, token_local].masked_fill(
            ~exists[token_view], torch.inf
        )
        neighbor = to_cells.scatter(1, own, torch.inf).argmin(1, keepdim=True)

        def contrast(term: Tensor) -> Tensor:
            rows_of_term = term[token_view, token_local]
            gap = rows_of_term.gather(1, neighbor) - rows_of_term.gather(1, own)
            return per_view_mean(gap[:, 0])

        token = squared(x, cells)
        spatial_scale, token_scale = contrast(spatial), contrast(token)

        for iteration in range(self.iterations):
            if iteration > 0:
                spatial, token = squared(positions, cells), squared(x, cells)
            reachable = exists.clone()
            empty = torch.bincount(cells, minlength=count) == 0
            reachable[cell_view[empty], cell_local[empty]] = False
            distance = (1 - mix) * spatial / spatial_scale + mix * token / token_scale
            distance.masked_fill_(~reachable[:, None], torch.inf)
            choice = distance.argmin(-1)[token_view, token_local]
            cells = cell_start[token_view] + choice
        return cells


def match_tracks(
    cells: Tensor,
    count: int,
    keys: Tensor,
    slots: Tensor,
    min_overlap: float,
    min_share: float,
) -> tuple[Tensor, int]:
    """Tracks of cells that hold the same nuclei in different views.

    Views are numbered within their sample, the 2 global views first. A track
    starts from a cell of the first global view and a cell of the second that
    are each other's best match by the Jaccard index of their nuclei, if it is
    at least `min_overlap`. A cell of a local view joins the track whose global
    cells hold the largest share of its nuclei, if that share is at least
    `min_share`. Local cells are small and often cut by the view border, so
    their Jaccard index with a whole track is low even inside it.

    Args:
        cells: Cell of each nucleus, [nuclei].
        count: Number of cells.
        keys: Identity of each nucleus, the same in every view of a sample and
            unique across samples, [nuclei].
        slots: View of each nucleus within its sample, [nuclei].
        min_overlap: Least Jaccard index of the global cells of a track.
        min_share: Least share of the nuclei of a local cell in its track.

    Returns:
        The track of each cell or -1, [count], and the number of tracks.
    """
    device = cells.device
    sizes = torch.bincount(cells, minlength=count).float()
    cell_slot = torch.zeros(count, dtype=torch.long, device=device)
    cell_slot[cells] = slots

    def finder(slot: int):  # the cell in `slot` of the nuclei with given keys
        inside = slots == slot
        order = keys[inside].argsort()
        sorted_keys, sorted_cells = keys[inside][order], cells[inside][order]

        def find(queries: Tensor) -> Tensor:
            if len(sorted_keys) == 0:
                return torch.full_like(queries, -1)
            at = torch.searchsorted(sorted_keys, queries).clamp(max=len(order) - 1)
            return torch.where(sorted_keys[at] == queries, sorted_cells[at], -1)

        return find

    def local_index(mask: Tensor) -> tuple[Tensor, Tensor]:
        members = mask.nonzero()[:, 0]
        index = torch.full((count,), -1, dtype=torch.long, device=device)
        index[members] = torch.arange(len(members), device=device)
        return members, index

    first, first_index = local_index(cell_slot == 0)
    second, second_index = local_index(cell_slot == 1)
    tracks = torch.full((count,), -1, dtype=torch.long, device=device)
    if len(first) == 0 or len(second) == 0:
        return tracks, 0
    in_first = slots == 0
    partner = finder(1)(keys[in_first])
    shared_nuclei = partner >= 0
    shared = torch.zeros(len(first), len(second), device=device)
    shared.index_put_(
        (
            first_index[cells[in_first][shared_nuclei]],
            second_index[partner[shared_nuclei]],
        ),
        torch.ones(int(shared_nuclei.sum()), device=device),
        accumulate=True,
    )
    union = sizes[first][:, None] + sizes[second][None] - shared
    jaccard = shared / union.clamp_min(1)
    best, best_back = jaccard.argmax(1), jaccard.argmax(0)
    rows = torch.arange(len(first), device=device)
    starts = (
        (best_back[best] == rows) & (jaccard[rows, best] >= min_overlap)
    ).nonzero()[:, 0]
    track_count = len(starts)
    if track_count == 0:
        return tracks, 0
    numbers = torch.arange(track_count, device=device)
    tracks[first[starts]] = numbers
    tracks[second[best[starts]]] = numbers

    local = slots >= 2
    local_cells, local_cell_index = local_index(cell_slot >= 2)
    if len(local_cells) == 0:
        return tracks, track_count
    row = local_cell_index[cells[local]]
    shares = torch.zeros(len(local_cells), track_count, device=device)
    via = []
    for slot in (0, 1):
        found = finder(slot)(keys[local])
        via.append(torch.where(found >= 0, tracks[found.clamp_min(0)], -1))
    # A nucleus in the global cells of a track counts once, through either view.
    for track in (via[0], torch.where(via[1] != via[0], via[1], -1)):
        hit = track >= 0
        shares.index_put_(
            (row[hit], track[hit]),
            torch.ones(int(hit.sum()), device=device),
            accumulate=True,
        )
    shares /= sizes[local_cells][:, None].clamp_min(1)
    value, joined = shares.max(1)
    tracks[local_cells[value >= min_share]] = joined[value >= min_share]
    return tracks, track_count
