import numpy as np
import pytest

from logicad_fig2.filtering import select_description


@pytest.mark.parametrize("count", [1, 2, 3])
def test_small_k_fallback(count):
    result = select_description([[1, 0]] * count, n_neighbors=20)
    assert 0 <= result["selected_index"] < count
    assert result["n_neighbors"] <= count - 1
    assert result["fallback"]


def test_lof_outlier_and_seed():
    matrix = [[1, 0], [1, .001], [1, -.001], [1, .002], [-1, 0]]
    result = select_description(matrix, seed=2)
    assert 4 not in result["inlier_indices"]
    assert result == select_description(matrix, seed=2)


def test_invalid_embedding_fails():
    with pytest.raises(ValueError):
        select_description([[np.nan, 1]])
