import pathlib
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hedgementation_utils.io.io_manager import HedgementationIOManager

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

IMG_W = 4
IMG_H = 4
NUM_X_BANDS = 10
NUM_X_CLOUD_BANDS = 2


def make_X_array(num_dates=1, num_bands=NUM_X_BANDS, h=IMG_H, w=IMG_W):
    return np.random.rand(num_dates * num_bands * h * w).astype(np.float32)


def make_y_array(h=IMG_H, w=IMG_W):
    return np.random.rand(1, h, w).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_AEF_BANDS = 64


def make_aef_array(num_bands=NUM_AEF_BANDS, h=IMG_H, w=IMG_W):
    return np.random.rand(num_bands, h, w).astype(np.float32)


@pytest.fixture
def dataset_root(tmp_path):
    for d in ("X", "y", "X_cloud", "y_id", "rpg", "AEF"):
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def manager(dataset_root):
    return HedgementationIOManager(
        dataset_root=str(dataset_root),
        img_width=IMG_W,
        img_height=IMG_H,
    )


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_dataset_root_stored(self, manager, dataset_root):
        assert manager.dataset_root == str(dataset_root)

    def test_default_subdirectory_paths(self, manager, dataset_root):
        assert manager.X_dir == pathlib.Path(dataset_root, "X")
        assert manager.y_dir == pathlib.Path(dataset_root, "y")
        assert manager.X_cloud_dir == pathlib.Path(dataset_root, "X_cloud")
        assert manager.y_id_dir == pathlib.Path(dataset_root, "y_id")
        assert manager.rpg_mask_dir == pathlib.Path(dataset_root, "rpg")
        assert manager.aef_dir == pathlib.Path(dataset_root, "AEF")

    def test_absolute_aef_dir_overrides_root(self, dataset_root, tmp_path):
        # AEF embeddings may live outside dataset_root; an absolute aef_dir wins.
        external = tmp_path / "elsewhere" / "AEF"
        m = HedgementationIOManager(
            str(dataset_root), aef_dir=str(external), img_width=IMG_W, img_height=IMG_H
        )
        assert m.aef_dir == external

    def test_custom_subdirectory_applied(self, dataset_root):
        (dataset_root / "images").mkdir()
        m = HedgementationIOManager(str(dataset_root), X_dir="images", img_width=IMG_W, img_height=IMG_H)
        assert m.X_dir == pathlib.Path(dataset_root, "images")

    def test_default_patterns(self, manager):
        assert manager.X_pattern == "X_{}"
        assert manager.y_pattern == "y_{}"
        assert manager.X_cloud_pattern == "X_cloud_{}"
        assert manager.y_id_pattern == "y_id_{}"
        assert manager.rpg_pattern == "rpg_{}"
        assert manager.aef_pattern == "AEF_{}"


    def test_default_band_counts(self, manager):
        assert manager.num_X_bands == 10
        assert manager.num_X_cloud_bands == 2

    def test_custom_band_counts(self, dataset_root):
        m = HedgementationIOManager(
            str(dataset_root), num_X_bands=5, num_X_cloud_bands=3,
            img_width=IMG_W, img_height=IMG_H
        )
        assert m.num_X_bands == 5
        assert m.num_X_cloud_bands == 3

    def test_img_dimensions_stored(self, manager):
        assert manager.img_width == IMG_W
        assert manager.img_height == IMG_H


# ---------------------------------------------------------------------------
# TestGetLoadPath
# ---------------------------------------------------------------------------

class TestGetLoadPath:
    def test_identifier_builds_path(self, manager):
        path = manager.get_load_path(
            identifier=7, load_path=None,
            file_type="npy", directory=manager.X_dir, pattern="X_{}"
        )
        assert path == pathlib.Path(manager.X_dir, "X_7.npy")

    def test_explicit_load_path_returned(self, manager, tmp_path):
        explicit = str(tmp_path / "custom.npy")
        path = manager.get_load_path(
            identifier=None, load_path=explicit,
            file_type="npy", directory=manager.X_dir, pattern="X_{}"
        )
        assert path == pathlib.Path(explicit)

    def test_both_raises(self, manager, tmp_path):
        with pytest.raises(ValueError):
            manager.get_load_path(
                identifier=1, load_path=str(tmp_path / "x.npy"),
                file_type="npy", directory=manager.X_dir, pattern="X_{}"
            )

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.get_load_path(
                identifier=None, load_path=None,
                file_type="npy", directory=manager.X_dir, pattern="X_{}"
            )

    def test_pattern_formatted_with_identifier(self, manager):
        path = manager.get_load_path(
            identifier=42, load_path=None,
            file_type="tif", directory=manager.X_dir, pattern="patch_{}"
        )
        assert path.name == "patch_42.tif"

    def test_file_type_included_in_suffix(self, manager):
        path = manager.get_load_path(
            identifier=0, load_path=None,
            file_type="tif", directory=manager.X_dir, pattern="X_{}"
        )
        assert path.suffix == ".tif"


# ---------------------------------------------------------------------------
# TestBandsToDates
# ---------------------------------------------------------------------------

class TestBandsToDates:
    def test_extracts_unique_dates(self, manager):
        bands = ["2021-01-01_B1", "2021-01-01_B2", "2021-06-15_B1", "2021-06-15_B2"]
        result = manager._bands_to_dates(bands, n_bands_per_timestep=2)
        assert set(result) == {"2021-01-01", "2021-06-15"}

    def test_result_is_sorted(self, manager):
        bands = ["2021-06-15_B1", "2021-01-01_B1", "2021-03-10_B1"]
        result = manager._bands_to_dates(bands, n_bands_per_timestep=2)
        assert list(result) == sorted(result)


# ---------------------------------------------------------------------------
# TestLoadX
# ---------------------------------------------------------------------------

class TestLoadX:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "X" / "X_0.npy", make_X_array())
        result = manager.load_X(identifier=0, file_type="npy")
        assert result.shape == (1, NUM_X_BANDS, IMG_H, IMG_W)

    def test_loads_npy_by_path(self, manager, dataset_root):
        path = dataset_root / "X" / "custom.npy"
        np.save(path, make_X_array())
        result = manager.load_X(load_path=str(path))
        assert result.shape == (1, NUM_X_BANDS, IMG_H, IMG_W)

    def test_multiple_dates(self, manager, dataset_root):
        np.save(dataset_root / "X" / "X_1.npy", make_X_array(num_dates=3))
        result = manager.load_X(identifier=1, file_type="npy")
        assert result.shape == (3, NUM_X_BANDS, IMG_H, IMG_W)

    def test_values_preserved(self, manager, dataset_root):
        data = np.arange(NUM_X_BANDS * IMG_H * IMG_W, dtype=np.float32)
        np.save(dataset_root / "X" / "X_2.npy", data)
        result = manager.load_X(identifier=2, file_type="npy")
        np.testing.assert_array_equal(result.reshape(-1), data)

    def test_file_type_override(self, manager, dataset_root):
        np.save(dataset_root / "X" / "X_0.npy", make_X_array())
        result = manager.load_X(identifier=0, file_type="npy")
        assert result.shape == (1, NUM_X_BANDS, IMG_H, IMG_W)

    def test_both_identifier_and_path_raises(self, manager, dataset_root):
        with pytest.raises(ValueError):
            manager.load_X(identifier=0, load_path=str(dataset_root / "X" / "x.npy"))

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_X()

    def test_loads_tif_return_as_dataset(self, manager, dataset_root):
        tif_path = dataset_root / "X" / "X_0.tif"
        tif_path.touch()
        mock_ds = MagicMock()
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_ds
        with patch("hedgementation_utils.io.io_manager.rasterio", mock_rasterio):
            result = manager.load_X(load_path=str(tif_path), return_as_dataset=True)
        assert result is mock_ds

    def test_loads_tif_data(self, manager, dataset_root):
        tif_path = dataset_root / "X" / "X_0.tif"
        tif_path.touch()
        raw = np.ones(NUM_X_BANDS * IMG_H * IMG_W, dtype=np.float32)
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value.__enter__.return_value.read.return_value = raw
        mock_rasterio.open.return_value.__exit__.return_value = False
        with patch("hedgementation_utils.io.io_manager.rasterio", mock_rasterio):
            result = manager.load_X(load_path=str(tif_path))
        assert result.shape == (1, NUM_X_BANDS, IMG_H, IMG_W)

    def test_loads_tif_with_dates(self, manager, dataset_root):
        tif_path = dataset_root / "X" / "X_0.tif"
        tif_path.touch()
        raw = np.ones(NUM_X_BANDS * IMG_H * IMG_W, dtype=np.float32)
        bands = [f"2021-01-01_B{i}" for i in range(NUM_X_BANDS)]
        mock_rasterio = MagicMock()
        mock_src = mock_rasterio.open.return_value.__enter__.return_value
        mock_src.read.return_value = raw
        mock_src.descriptions = bands
        mock_rasterio.open.return_value.__exit__.return_value = False
        with patch("hedgementation_utils.io.io_manager.rasterio", mock_rasterio):
            result, dates = manager.load_X(load_path=str(tif_path), return_dates=True)
        assert result.shape == (1, NUM_X_BANDS, IMG_H, IMG_W)
        assert "2021-01-01" in list(dates)


# ---------------------------------------------------------------------------
# TestLoadXCloud
# ---------------------------------------------------------------------------

class TestLoadXCloud:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "X_cloud" / "X_cloud_0.npy", make_X_array(num_bands=NUM_X_CLOUD_BANDS))
        result = manager.load_X_cloud(identifier=0, file_type="npy")
        assert result.shape == (1, NUM_X_CLOUD_BANDS, IMG_H, IMG_W)

    def test_multiple_dates(self, manager, dataset_root):
        np.save(dataset_root / "X_cloud" / "X_cloud_5.npy", make_X_array(num_dates=2, num_bands=NUM_X_CLOUD_BANDS))
        result = manager.load_X_cloud(identifier=5, file_type="npy")
        assert result.shape == (2, NUM_X_CLOUD_BANDS, IMG_H, IMG_W)

    def test_loads_by_path(self, manager, dataset_root):
        path = dataset_root / "X_cloud" / "custom.npy"
        np.save(path, make_X_array(num_bands=NUM_X_CLOUD_BANDS))
        result = manager.load_X_cloud(load_path=str(path))
        assert result.shape == (1, NUM_X_CLOUD_BANDS, IMG_H, IMG_W)

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_X_cloud()


# ---------------------------------------------------------------------------
# TestLoadY
# ---------------------------------------------------------------------------

class TestLoadY:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "y" / "y_0.npy", make_y_array())
        result = manager.load_y(identifier=0, file_type="npy")
        assert result.shape == (IMG_H, IMG_W)

    def test_loads_npy_by_path(self, manager, dataset_root):
        path = dataset_root / "y" / "custom_y.npy"
        np.save(path, make_y_array())
        result = manager.load_y(load_path=str(path))
        assert result.shape == (IMG_H, IMG_W)

    def test_squeeze_removes_channel_dim(self, manager, dataset_root):
        np.save(dataset_root / "y" / "y_1.npy", np.ones((1, IMG_H, IMG_W), dtype=np.float32))
        result = manager.load_y(identifier=1, file_type="npy")
        assert result.ndim == 2

    def test_values_preserved(self, manager, dataset_root):
        data = np.arange(IMG_H * IMG_W, dtype=np.float32).reshape(1, IMG_H, IMG_W)
        np.save(dataset_root / "y" / "y_2.npy", data)
        result = manager.load_y(identifier=2, file_type="npy")
        np.testing.assert_array_equal(result, data.squeeze())

    def test_both_identifier_and_path_raises(self, manager, dataset_root):
        with pytest.raises(ValueError):
            manager.load_y(identifier=0, load_path=str(dataset_root / "y" / "y.npy"))

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_y()

    def test_loads_tif(self, manager, dataset_root):
        tif_path = dataset_root / "y" / "y_0.tif"
        tif_path.touch()
        raw = np.ones((1, IMG_H, IMG_W), dtype=np.float32)
        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value.__enter__.return_value.read.return_value = raw
        mock_rasterio.open.return_value.__exit__.return_value = False
        with patch("hedgementation_utils.io.io_manager.rasterio", mock_rasterio):
            result = manager.load_y(load_path=str(tif_path))
        assert result.shape == (IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# TestLoadYId
# ---------------------------------------------------------------------------

class TestLoadYId:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "y_id" / "y_id_0.npy", make_y_array())
        result = manager.load_y_id(identifier=0, file_type="npy")
        assert result.shape == (IMG_H, IMG_W)

    def test_squeeze_removes_channel_dim(self, manager, dataset_root):
        np.save(dataset_root / "y_id" / "y_id_1.npy", np.ones((1, IMG_H, IMG_W), dtype=np.float32))
        result = manager.load_y_id(identifier=1, file_type="npy")
        assert result.ndim == 2

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_y_id()


# ---------------------------------------------------------------------------
# TestLoadRpgMask
# ---------------------------------------------------------------------------

class TestLoadRpgMask:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "rpg" / "rpg_0.npy", make_y_array())
        result = manager.load_rpg_mask(identifier=0)
        assert result.shape == (IMG_H, IMG_W)

    def test_loads_npy_by_path(self, manager, dataset_root):
        path = dataset_root / "rpg" / "custom_rpg.npy"
        np.save(path, make_y_array())
        result = manager.load_rpg_mask(load_path=str(path))
        assert result.shape == (IMG_H, IMG_W)

    def test_values_preserved(self, manager, dataset_root):
        data = (np.random.rand(IMG_H, IMG_W) > 0.5).astype(np.float32).reshape(1, IMG_H, IMG_W)
        np.save(dataset_root / "rpg" / "rpg_7.npy", data)
        result = manager.load_rpg_mask(identifier=7)
        np.testing.assert_array_equal(result, data.squeeze())

    def test_squeeze_removes_channel_dim(self, manager, dataset_root):
        np.save(dataset_root / "rpg" / "rpg_3.npy", np.ones((1, IMG_H, IMG_W), dtype=np.float32))
        result = manager.load_rpg_mask(identifier=3)
        assert result.ndim == 2

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_rpg_mask()


# ---------------------------------------------------------------------------
# TestLoadAef
# ---------------------------------------------------------------------------

class TestLoadAef:
    def test_loads_npy_by_identifier(self, manager, dataset_root):
        np.save(dataset_root / "AEF" / "AEF_0.npy", make_aef_array())
        result = manager.load_aef(identifier=0)
        assert result.shape == (NUM_AEF_BANDS, IMG_H, IMG_W)

    def test_loads_npy_by_path(self, manager, dataset_root):
        path = dataset_root / "AEF" / "custom_aef.npy"
        np.save(path, make_aef_array())
        result = manager.load_aef(load_path=str(path))
        assert result.shape == (NUM_AEF_BANDS, IMG_H, IMG_W)

    def test_not_reshaped_or_squeezed(self, manager, dataset_root):
        # A singleton band dim must survive (unlike load_y, which squeezes).
        data = np.ones((1, IMG_H, IMG_W), dtype=np.float32)
        np.save(dataset_root / "AEF" / "AEF_9.npy", data)
        result = manager.load_aef(identifier=9)
        assert result.shape == (1, IMG_H, IMG_W)

    def test_values_preserved(self, manager, dataset_root):
        data = np.arange(NUM_AEF_BANDS * IMG_H * IMG_W, dtype=np.float32).reshape(
            NUM_AEF_BANDS, IMG_H, IMG_W
        )
        np.save(dataset_root / "AEF" / "AEF_2.npy", data)
        result = manager.load_aef(identifier=2)
        np.testing.assert_array_equal(result, data)

    def test_both_identifier_and_path_raises(self, manager, dataset_root):
        with pytest.raises(ValueError):
            manager.load_aef(identifier=0, load_path=str(dataset_root / "AEF" / "a.npy"))

    def test_neither_raises(self, manager):
        with pytest.raises(ValueError):
            manager.load_aef()


# ---------------------------------------------------------------------------
# Helpers for the date-alignment / cloud-filter test classes below
# ---------------------------------------------------------------------------

def make_X(num_dates, num_bands=NUM_X_BANDS, fill=None):
    shape = (num_dates, num_bands, IMG_H, IMG_W)
    if fill is None:
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return np.full(shape, fill, dtype=np.float32)


def make_X_cloud(per_timestep_means, num_bands=NUM_X_CLOUD_BANDS, band_ind=0):
    """Build a cloud array whose `band_ind` slice has the requested per-timestep mean."""
    T = len(per_timestep_means)
    arr = np.zeros((T, num_bands, IMG_H, IMG_W), dtype=np.float32)
    for t, m in enumerate(per_timestep_means):
        arr[t, band_ind, :, :] = m
    return arr


# ---------------------------------------------------------------------------
# TestLoadXNpyReturnDates - new error path
# ---------------------------------------------------------------------------

class TestLoadXNpyReturnDates:
    def test_npy_with_return_dates_raises(self, manager, dataset_root):
        np.save(dataset_root / "X" / "X_0.npy", make_X_array())
        with pytest.raises(ValueError, match="return_dates"):
            manager.load_X(identifier=0, file_type="npy", return_dates=True)

    def test_npy_cloud_with_return_dates_raises(self, manager, dataset_root):
        np.save(
            dataset_root / "X_cloud" / "X_cloud_0.npy",
            make_X_array(num_bands=NUM_X_CLOUD_BANDS),
        )
        with pytest.raises(ValueError, match="return_dates"):
            manager.load_X_cloud(identifier=0, file_type="npy", return_dates=True)


# ---------------------------------------------------------------------------
# TestGetMeanCloudScores
# ---------------------------------------------------------------------------

class TestGetMeanCloudScores:
    def test_returns_one_value_per_timestep(self, manager):
        X_cloud = make_X_cloud([0.0, 0.0, 0.0])
        out = manager._get_mean_cloud_scores(X_cloud)
        assert out.shape == (3,)

    def test_mean_computed_per_timestep(self, manager):
        X_cloud = make_X_cloud([0.1, 0.5, 0.9])
        out = manager._get_mean_cloud_scores(X_cloud)
        np.testing.assert_allclose(out, [0.1, 0.5, 0.9])

    def test_band_ind_selects_band(self, manager):
        X_cloud = np.zeros((2, NUM_X_CLOUD_BANDS, IMG_H, IMG_W), dtype=np.float32)
        X_cloud[:, 0, :, :] = 0.1
        X_cloud[:, 1, :, :] = 0.7
        out = manager._get_mean_cloud_scores(X_cloud, band_ind=1)
        np.testing.assert_allclose(out, [0.7, 0.7])

    def test_mean_across_spatial_dims(self, manager):
        X_cloud = np.zeros((1, NUM_X_CLOUD_BANDS, IMG_H, IMG_W), dtype=np.float32)
        X_cloud[0, 0, 0, 0] = 1.0  # one pixel of 16 set; mean = 1/16
        out = manager._get_mean_cloud_scores(X_cloud)
        np.testing.assert_allclose(out, [1.0 / (IMG_H * IMG_W)])


# ---------------------------------------------------------------------------
# TestLimitAboveCloudThreshold
# ---------------------------------------------------------------------------

class TestLimitAboveCloudThreshold:
    # Cloud Score+ semantics: higher score = clearer, lower score = cloudier.
    # `_limit_above_cloud_threshold` keeps timesteps whose mean score > threshold,
    # i.e. those whose clarity exceeds the minimum required clarity.

    def test_keeps_clear_timesteps(self, manager):
        X = make_X(num_dates=3)
        # Under CS+: 0.5 is clear, 0.05/0.1 are cloudy; with threshold 0.2 keep only the clear one.
        X_cloud = make_X_cloud([0.05, 0.5, 0.1])
        X_out, cloud_out = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, threshold=0.2
        )
        assert X_out.shape[0] == 1
        np.testing.assert_array_equal(X_out, X[[1]])
        np.testing.assert_array_equal(cloud_out, X_cloud[[1]])

    def test_drops_all_when_all_cloudy(self, manager):
        X = make_X(num_dates=3)
        # All-low scores under CS+ means all cloudy -> none clear enough.
        X_cloud = make_X_cloud([0.0, 0.05, 0.1])
        X_out, cloud_out = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, threshold=0.2
        )
        assert X_out.shape == (0, NUM_X_BANDS, IMG_H, IMG_W)
        assert cloud_out.shape == (0, NUM_X_CLOUD_BANDS, IMG_H, IMG_W)

    def test_keeps_all_when_all_clear(self, manager):
        X = make_X(num_dates=4)
        # All-high scores under CS+ means all clear -> all kept.
        X_cloud = make_X_cloud([0.5, 0.7, 0.9, 1.0])
        X_out, cloud_out = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, threshold=0.2
        )
        assert X_out.shape[0] == 4
        np.testing.assert_array_equal(X_out, X)

    def test_dates_returned_when_passed(self, manager):
        X = make_X(num_dates=3)
        X_cloud = make_X_cloud([0.5, 0.05, 0.9])
        dates = pd.Series(["2021-01-01", "2021-02-01", "2021-03-01"])
        X_out, cloud_out, dates_out = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, dates=dates, threshold=0.2
        )
        # First and third are clear; middle is cloudy.
        assert list(dates_out) == ["2021-01-01", "2021-03-01"]
        assert X_out.shape[0] == len(dates_out)

    def test_returns_two_tuple_when_no_dates(self, manager):
        X = make_X(num_dates=2)
        X_cloud = make_X_cloud([0.9, 0.9])
        out = manager._limit_above_cloud_threshold(X=X, X_cloud=X_cloud, threshold=0.2)
        assert len(out) == 2

    def test_band_ind_respected(self, manager):
        X = make_X(num_dates=3)
        X_cloud = np.zeros((3, NUM_X_CLOUD_BANDS, IMG_H, IMG_W), dtype=np.float32)
        X_cloud[:, 0, :, :] = 0.05  # band 0 is "cloudy" for all under CS+
        X_cloud[:, 1, :, :] = np.array([0.9, 0.05, 0.9])[:, None, None]  # band 1 mixed
        X_out, _ = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, threshold=0.2, band_ind=1
        )
        # Using band 1: timesteps 0 and 2 are clear, middle is cloudy.
        assert X_out.shape[0] == 2
        np.testing.assert_array_equal(X_out, X[[0, 2]])

    def test_threshold_is_strict_greater_than(self, manager):
        # Equal-to-threshold should be dropped (strict >).
        X = make_X(num_dates=2)
        X_cloud = make_X_cloud([0.2, 0.3])
        X_out, _ = manager._limit_above_cloud_threshold(
            X=X, X_cloud=X_cloud, threshold=0.2
        )
        assert X_out.shape[0] == 1
        np.testing.assert_array_equal(X_out, X[[1]])


# ---------------------------------------------------------------------------
# TestFilterToConsensus
# ---------------------------------------------------------------------------

class TestFilterToConsensus:
    def test_simple_intersection(self, manager):
        dates = pd.Series(["2021-01-01", "2021-02-01", "2021-03-01"])
        counter = Counter({"2021-01-01": 1, "2021-03-01": 1})
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == [0, 2]

    def test_date_not_in_consensus_excluded(self, manager):
        dates = pd.Series(["2021-01-01", "2021-02-01"])
        counter = Counter({"2021-01-01": 1})
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == [0]

    def test_duplicates_keeps_first_k_occurrences(self, manager):
        # date appears 3 times in series but only 2 allowed in consensus
        dates = pd.Series(["2021-01-01", "2021-01-01", "2021-01-01", "2021-02-01"])
        counter = Counter({"2021-01-01": 2, "2021-02-01": 1})
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == [0, 1, 3]

    def test_empty_consensus_returns_empty(self, manager):
        dates = pd.Series(["2021-01-01", "2021-02-01"])
        counter = Counter()
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == []

    def test_all_in_consensus(self, manager):
        dates = pd.Series(["2021-01-01", "2021-02-01", "2021-03-01"])
        counter = Counter(dates)
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == [0, 1, 2]

    def test_consensus_count_greater_than_series_count(self, manager):
        # Consensus allows more than the series has — should keep all available
        dates = pd.Series(["2021-01-01"])
        counter = Counter({"2021-01-01": 5})
        inds = manager._filter_to_consensus(dates, counter)
        assert inds == [0]


# ---------------------------------------------------------------------------
# TestFindConsensusDates
# ---------------------------------------------------------------------------

class TestFindConsensusDates:
    def test_identical_series(self, manager):
        d1 = pd.Series(["2021-01-01", "2021-02-01"])
        d2 = pd.Series(["2021-01-01", "2021-02-01"])
        consensus, i1, i2 = manager._find_consensus_dates(d1, d2)
        assert list(consensus) == ["2021-01-01", "2021-02-01"]
        assert i1 == [0, 1]
        assert i2 == [0, 1]

    def test_disjoint_series(self, manager):
        d1 = pd.Series(["2021-01-01"])
        d2 = pd.Series(["2022-01-01"])
        consensus, i1, i2 = manager._find_consensus_dates(d1, d2)
        assert list(consensus) == []
        assert i1 == []
        assert i2 == []

    def test_partial_overlap(self, manager):
        d1 = pd.Series(["2021-01-01", "2021-02-01", "2021-03-01"])
        d2 = pd.Series(["2021-02-01", "2021-03-01", "2021-04-01"])
        consensus, i1, i2 = manager._find_consensus_dates(d1, d2)
        assert list(consensus) == ["2021-02-01", "2021-03-01"]
        assert i1 == [1, 2]
        assert i2 == [0, 1]

    def test_duplicates_intersected_by_min_count(self, manager):
        d1 = pd.Series(["2021-01-01", "2021-01-01", "2021-02-01"])
        d2 = pd.Series(["2021-01-01", "2021-02-01", "2021-02-01"])
        consensus, i1, i2 = manager._find_consensus_dates(d1, d2)
        # min of {2021-01-01: 2 vs 1} = 1; min of {2021-02-01: 1 vs 2} = 1
        assert list(consensus) == ["2021-01-01", "2021-02-01"]
        assert i1 == [0, 2]  # first 2021-01-01 + only 2021-02-01
        assert i2 == [0, 1]  # only 2021-01-01 + first 2021-02-01

    def test_return_inds_false_returns_series_only(self, manager):
        d1 = pd.Series(["2021-01-01", "2021-02-01"])
        d2 = pd.Series(["2021-02-01"])
        result = manager._find_consensus_dates(d1, d2, return_inds=False)
        assert isinstance(result, pd.Series)
        assert list(result) == ["2021-02-01"]

    def test_consensus_sorted(self, manager):
        # Even if inputs are sorted, multi-date duplicate intersections must be sorted
        d1 = pd.Series(["2021-03-01", "2021-01-01", "2021-02-01"])
        d2 = pd.Series(["2021-01-01", "2021-02-01", "2021-03-01"])
        consensus = manager._find_consensus_dates(d1, d2, return_inds=False)
        assert list(consensus) == sorted(list(consensus))


# ---------------------------------------------------------------------------
# TestLoadXDataDateAligned
# ---------------------------------------------------------------------------

def _patch_loads(manager, X_data, X_dates, cloud_data, cloud_dates):
    """Return two context-manager patches that stub load_X and load_X_cloud."""
    p1 = patch.object(manager, "load_X", return_value=(X_data, pd.Series(X_dates)))
    p2 = patch.object(
        manager, "load_X_cloud", return_value=(cloud_data, pd.Series(cloud_dates))
    )
    return p1, p2


class TestLoadXDataDateAligned:
    def test_identical_dates_passes_through(self, manager):
        dates = ["2021-01-01", "2021-02-01"]
        X = make_X(2)
        cloud = make_X_cloud([0.0, 0.0])
        p1, p2 = _patch_loads(manager, X, dates, cloud, dates)
        with p1, p2:
            X_out, cloud_out, consensus = manager.load_X_data_date_aligned(identifier=0)
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(cloud_out, cloud)
        assert list(consensus) == dates

    def test_X_has_extra_date_is_filtered(self, manager):
        X_dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        cloud_dates = ["2021-01-01", "2021-03-01"]
        X = make_X(3)
        cloud = make_X_cloud([0.0, 0.0])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, cloud_out, consensus = manager.load_X_data_date_aligned(identifier=0)
        assert list(consensus) == ["2021-01-01", "2021-03-01"]
        np.testing.assert_array_equal(X_out, X[[0, 2]])
        np.testing.assert_array_equal(cloud_out, cloud)

    def test_cloud_has_extra_date_is_filtered(self, manager):
        X_dates = ["2021-01-01", "2021-02-01"]
        cloud_dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        X = make_X(2)
        cloud = make_X_cloud([0.0, 0.0, 0.0])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, cloud_out, consensus = manager.load_X_data_date_aligned(identifier=0)
        assert list(consensus) == ["2021-01-01", "2021-02-01"]
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(cloud_out, cloud[[0, 1]])

    def test_no_overlap_returns_empty(self, manager):
        X_dates = ["2021-01-01"]
        cloud_dates = ["2022-01-01"]
        X = make_X(1)
        cloud = make_X_cloud([0.0])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, cloud_out, consensus = manager.load_X_data_date_aligned(identifier=0)
        assert X_out.shape[0] == 0
        assert cloud_out.shape[0] == 0
        assert list(consensus) == []

    def test_duplicate_dates_aligned(self, manager):
        # X has two occurrences of 2021-01-01, cloud has only one
        X_dates = ["2021-01-01", "2021-01-01", "2021-02-01"]
        cloud_dates = ["2021-01-01", "2021-02-01"]
        X = make_X(3)
        cloud = make_X_cloud([0.0, 0.0])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, cloud_out, consensus = manager.load_X_data_date_aligned(identifier=0)
        assert list(consensus) == ["2021-01-01", "2021-02-01"]
        # X keeps only first 2021-01-01 occurrence + the 2021-02-01
        np.testing.assert_array_equal(X_out, X[[0, 2]])
        np.testing.assert_array_equal(cloud_out, cloud)

    def test_return_dates_in_kwargs_silently_dropped(self, manager):
        # Caller accidentally includes return_dates in X_kwargs — shouldn't blow up
        dates = ["2021-01-01"]
        X = make_X(1)
        cloud = make_X_cloud([0.0])
        p1, p2 = _patch_loads(manager, X, dates, cloud, dates)
        with p1, p2:
            X_out, _, _ = manager.load_X_data_date_aligned(
                identifier=0,
                X_kwargs={"return_dates": False},
                X_cloud_kwargs={"return_dates": False},
            )
        assert X_out.shape[0] == 1

    def test_passes_through_other_kwargs(self, manager):
        dates = ["2021-01-01"]
        X = make_X(1)
        cloud = make_X_cloud([0.0])
        p1, p2 = _patch_loads(manager, X, dates, cloud, dates)
        with p1, p2:
            manager.load_X_data_date_aligned(
                identifier=0,
                X_kwargs={"file_type": "tif"},
                X_cloud_kwargs={"file_type": "tif"},
            )
            manager.load_X.assert_called_once()
            call_kwargs = manager.load_X.call_args.kwargs
            assert call_kwargs.get("file_type") == "tif"
            assert call_kwargs.get("return_dates") is True


# ---------------------------------------------------------------------------
# TestLoadXDataWithCloudThreshold
# ---------------------------------------------------------------------------

class TestLoadXDataWithCloudThreshold:
    # Cloud Score+ semantics: high score = clear, low score = cloudy.
    # Filter keeps timesteps with score > threshold.

    def test_end_to_end_filters_clouds_and_aligns(self, manager):
        X_dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        cloud_dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        X = make_X(3)
        # Timesteps 0 and 2 cloudy; timestep 1 clear.
        cloud = make_X_cloud([0.05, 0.9, 0.05])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, cloud_out, dates_out = manager.load_X_data_with_cloud_threshold(
                identifier=0, cloud_threshold=0.2
            )
        assert list(dates_out) == ["2021-02-01"]
        assert X_out.shape[0] == 1
        assert cloud_out.shape[0] == 1

    def test_uses_default_threshold_when_none(self, manager):
        manager.default_cloud_threshold = 0.5
        X_dates = ["2021-01-01", "2021-02-01"]
        X = make_X(2)
        # 0.3 is below default-0.5 (cloudy), 0.7 is above (clear).
        cloud = make_X_cloud([0.3, 0.7])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, X_dates)
        with p1, p2:
            _, _, dates_out = manager.load_X_data_with_cloud_threshold(identifier=0)
        assert list(dates_out) == ["2021-02-01"]

    def test_explicit_threshold_zero_keeps_any_positive_score(self, manager):
        # With strict > and threshold=0.0, scores of exactly 0 are dropped,
        # any positive score is kept.
        X_dates = ["2021-01-01", "2021-02-01"]
        X = make_X(2)
        cloud = make_X_cloud([0.0, 0.01])
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, X_dates)
        with p1, p2:
            X_out, _, dates_out = manager.load_X_data_with_cloud_threshold(
                identifier=0, cloud_threshold=0.0
            )
        assert X_out.shape[0] == 1
        assert list(dates_out) == ["2021-02-01"]

    def test_explicit_band_zero_does_not_fall_back(self, manager):
        # Verify that passing band=0 explicitly is honored and does NOT silently
        # fall back to the instance default (here deliberately set to a different band).
        manager.default_cloud_filtering_band = 1
        X_dates = ["2021-01-01", "2021-02-01"]
        X = make_X(2)
        # Band 0 is clear (high CS+ score); band 1 is cloudy (low CS+ score).
        cloud = np.zeros((2, NUM_X_CLOUD_BANDS, IMG_H, IMG_W), dtype=np.float32)
        cloud[:, 0, :, :] = 0.9
        cloud[:, 1, :, :] = 0.05
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, X_dates)
        with p1, p2:
            X_out, _, _ = manager.load_X_data_with_cloud_threshold(
                identifier=0, cloud_threshold=0.2, band=0
            )
        # If band=0 is respected, all timesteps pass; if it fell back to band 1, none would.
        assert X_out.shape[0] == 2

    def test_band_default_applied_when_none(self, manager):
        manager.default_cloud_filtering_band = 1
        X_dates = ["2021-01-01"]
        X = make_X(1)
        # Default band is 1, which here is cloudy (low CS+ score) -> should be dropped.
        cloud = np.zeros((1, NUM_X_CLOUD_BANDS, IMG_H, IMG_W), dtype=np.float32)
        cloud[:, 0, :, :] = 0.9  # band 0 clear, but not the default
        cloud[:, 1, :, :] = 0.05
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, X_dates)
        with p1, p2:
            X_out, _, _ = manager.load_X_data_with_cloud_threshold(
                identifier=0, cloud_threshold=0.2
            )
        assert X_out.shape[0] == 0

    def test_dates_aligned_when_x_and_cloud_differ(self, manager):
        # X has dates A, B, C; cloud has A, C only — first align, then threshold.
        X_dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        cloud_dates = ["2021-01-01", "2021-03-01"]
        X = make_X(3)
        cloud = make_X_cloud([0.9, 0.05])  # A clear, C cloudy
        p1, p2 = _patch_loads(manager, X, X_dates, cloud, cloud_dates)
        with p1, p2:
            X_out, _, dates_out = manager.load_X_data_with_cloud_threshold(
                identifier=0, cloud_threshold=0.2
            )
        # After alignment X reduces to indices [0, 2] (dates A, C),
        # after threshold only A remains (C is cloudy under CS+).
        assert list(dates_out) == ["2021-01-01"]
        np.testing.assert_array_equal(X_out, X[[0]])
