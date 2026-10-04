import numpy as np
import pytest

from feature_ecdf import area_weights, spatial_ecdfs, union_area, weighted_ecdf


def test_union_avoids_duplicate_and_overlap_double_counting():
    assert union_area([[0, 0, 2, 2], [1, 0, 3, 2], [0, 0, 2, 2]]) == 6
    assert union_area([[0, 0, 1, 1], [3, 0, 4, 1]]) == 2


def test_clipped_valid_cells_and_class_overlap():
    weights, valid = area_weights([0, 1, 2], [0, 1], [0.5, 0, 2, 1], [[0, 0, 1, 1]], [[0.75, 0, 1.5, 1]])
    assert valid.all()
    np.testing.assert_allclose(weights[:, 0, :], [[1, 0], [0.5, 0.5], [0, 0.5]])


def test_region_normalization_and_shared_scale():
    curves = spatial_ecdfs(np.array([[[1, 2]]]), np.ones((1, 2)), [0, 1, 2], [0, 1], [0, 0, 2, 1],
                           [[0, 0, 0.5, 1]], [[1, 0, 2, 1]])
    assert set(curves) == {"Drone", "Bird", "Background"}
    assert curves["Bird"]["relative_rms"][0] == pytest.approx(2 * curves["Drone"]["relative_rms"][0])
    assert all(curve["cdf"][-1] == 1 for curve in curves.values())
    assert curves["Drone"]["total_weight"] == 0.5


def test_empty_regions_and_tied_values():
    assert weighted_ecdf([1, 2], [0, 0]) is None
    result = weighted_ecdf([2, 1, 1], [1, 1, 2])
    assert result["relative_rms"] == [1, 2]
    assert result["cdf"] == [0.75, 1]
