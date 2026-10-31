"""Regression for candidate positions that collide at graph-builder precision."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
from degraph import build_spatial_graph

from nfm.data.augmentations import Compose
from nfm.data.datasets.nuclei_dataset import NucleiDataset
from nfm.data.geometry import radial_polygons


def test_candidates_are_unique_after_physical_coordinates_become_float32():
    # The first two pixel positions are distinct float32 values. Scaling in
    # float64 keeps them distinct at 1e-4 um, but float32 MPP multiplication
    # merges them. Exercise the production arithmetic, including MPP precision.
    pixels = np.array(
        [
            [16000.001953125, 20000],
            [16000.0029296875, 20000],
            [16010, 20000],
            [16000, 20010],
            [16010, 20010],
        ],
        dtype=np.float32,
    )
    mpp = np.array([0.3, 0.3], dtype=np.float64)
    physical = pixels.astype(np.float64) * mpp
    assert len(np.unique(np.round(physical / 1e-4).astype(np.int64), axis=0)) == 5
    assert len(np.unique(pixels * mpp.astype(np.float32), axis=0)) == 4
    radii = np.full((5, 16), 2.0)
    ids = np.arange(41, 46, dtype=np.uint64)
    table = pa.table(
        {"points": pixels.tolist(), "radial_distances": radii.tolist(), "_rowid": ids}
    )
    slide = pd.Series(
        {
            "id": "precision",
            "organ": "test",
            "dataset": "test",
            "mpp_x": 0.3,
            "mpp_y": 0.3,
        }
    )
    dataset = NucleiDataset(
        pd.DataFrame([slide]),
        "unused",
        crop_size=4,
        efd_order=4,
        augmentation=Compose([]),
    )
    with patch("nfm.data.datasets.nuclei_dataset.lance.dataset") as open_lance:
        store = open_lance.return_value
        store.count_rows.return_value = 5
        store.to_table.return_value = table
        store.sample.return_value = pa.table({"points": [pixels[1].tolist()]})
        points, polygons, kept_ids, seed = dataset._candidates(slide)
        assert points.dtype == np.float32
        assert len(points) == len(polygons) == len(kept_ids) == 4
        assert len(np.unique(points, axis=0)) == len(points)
        assert set(kept_ids) == {41, 43, 44, 45}
        assert kept_ids[seed] == 41
        # Graph construction succeeds and still has one vertex per retained ID.
        assert len(build_spatial_graph(points)) == len(kept_ids)
        expected_polygons = radial_polygons(pixels, radii) * mpp.astype(np.float32)
        for polygon, nucleus_id in zip(polygons, kept_ids, strict=True):
            np.testing.assert_array_equal(
                polygon, expected_polygons[int(nucleus_id) - 41]
            )
        sample = dataset[0]
    for view_ids in sample["indices"]:
        assert set(view_ids) == {41, 43, 44, 45}
    np.testing.assert_array_equal(sample["matched_ids"], [41, 43, 44, 45])
