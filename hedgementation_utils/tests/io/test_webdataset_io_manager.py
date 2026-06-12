import io
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile

from hedgementation_utils.io.webdataset_io_manager import WDSHedgementationIOManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H, W = 4, 4
NUM_X_BANDS = 3
NUM_X_CLOUD_BANDS = 2
NUM_AEF_BANDS = 64
NUM_TIMESTEPS = 2
SAMPLE_ID = "42"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tif_bytes(n_bands, height=H, width=W, dtype="float32", descriptions=None):
    data = np.random.rand(n_bands, height, width).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mem = MemoryFile()
        with mem.open(driver="GTiff", count=n_bands, height=height, width=width, dtype=dtype) as dst:
            dst.write(data)
            if descriptions:
                for i, desc in enumerate(descriptions, 1):
                    dst.set_band_description(i, desc)
        mem.seek(0)
        return mem.read()


def _x_descriptions(num_timesteps=NUM_TIMESTEPS, num_bands=NUM_X_BANDS):
    return [
        f"2021-0{t + 1}-01_B{b}"
        for t in range(num_timesteps)
        for b in range(num_bands)
    ]


def _x_cloud_descriptions(num_timesteps=NUM_TIMESTEPS, num_bands=NUM_X_CLOUD_BANDS):
    return [
        f"2021-0{t + 1}-01_cloud{b}"
        for t in range(num_timesteps)
        for b in range(num_bands)
    ]


def make_aef_bytes(num_bands=NUM_AEF_BANDS, height=H, width=W):
    data = np.random.rand(num_bands, height, width).astype("float32")
    buf = io.BytesIO()
    np.save(buf, data)
    return buf.getvalue()


def make_sample(identifier=SAMPLE_ID, num_timesteps=NUM_TIMESTEPS):
    return {
        "__key__": identifier,
        "x.tif": make_tif_bytes(
            num_timesteps * NUM_X_BANDS,
            descriptions=_x_descriptions(num_timesteps),
        ),
        "x_cloud.tif": make_tif_bytes(
            num_timesteps * NUM_X_CLOUD_BANDS,
            descriptions=_x_cloud_descriptions(num_timesteps),
        ),
        "y.tif": make_tif_bytes(1),
        "y_id.tif": make_tif_bytes(1, dtype="int32"),
        "rpg_parcelles.tif": make_tif_bytes(1),
        "aef.npy": make_aef_bytes(),
    }


@pytest.fixture
def manager():
    sample = make_sample()
    with patch("hedgementation_utils.io.webdataset_io_manager.wds.WebDataset") as mock_wds:
        mock_wds.return_value = [sample]
        mgr = WDSHedgementationIOManager(
            dataset_root="/fake/path.tar",
            num_X_bands=NUM_X_BANDS,
            num_X_cloud_bands=NUM_X_CLOUD_BANDS,
            img_width=W,
            img_height=H,
        )
    return mgr


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_index_built_from_dataset(self):
        sample = make_sample("7")
        with patch("hedgementation_utils.io.webdataset_io_manager.wds.WebDataset") as mock_wds:
            mock_wds.return_value = [sample]
            mgr = WDSHedgementationIOManager("/fake.tar")
        assert "7" in mgr._index

    def test_multiple_samples_indexed(self):
        samples = [make_sample(str(i)) for i in range(3)]
        with patch("hedgementation_utils.io.webdataset_io_manager.wds.WebDataset") as mock_wds:
            mock_wds.return_value = samples
            mgr = WDSHedgementationIOManager("/fake.tar")
        assert set(mgr._index.keys()) == {"0", "1", "2"}

    def test_lookup_missing_raises_key_error(self, manager):
        with pytest.raises(KeyError, match="999"):
            manager._lookup(999)

    def test_lookup_int_identifier_converted_to_str(self, manager):
        sample = manager._lookup(42)
        assert sample["__key__"] == SAMPLE_ID


# ---------------------------------------------------------------------------
# TestLoadX
# ---------------------------------------------------------------------------

class TestLoadX:
    def test_shape(self, manager):
        X = manager.load_X(identifier=42)
        assert X.shape == (NUM_TIMESTEPS, NUM_X_BANDS, H, W)

    def test_no_reshape(self, manager):
        X = manager.load_X(identifier=42, reshape=False)
        assert X.shape == (NUM_TIMESTEPS * NUM_X_BANDS, H, W)

    def test_return_dates(self, manager):
        X, dates = manager.load_X(identifier=42, return_dates=True)
        assert X.shape == (NUM_TIMESTEPS, NUM_X_BANDS, H, W)
        assert isinstance(dates, pd.Series)
        assert len(dates) == NUM_TIMESTEPS
        assert "2021-01-01" in list(dates)
        assert "2021-02-01" in list(dates)

    def test_return_as_dataset_raises(self, manager):
        with pytest.raises(NotImplementedError):
            manager.load_X(identifier=42, return_as_dataset=True)

    def test_missing_identifier_raises(self, manager):
        with pytest.raises(KeyError):
            manager.load_X(identifier=999)


# ---------------------------------------------------------------------------
# TestLoadXCloud
# ---------------------------------------------------------------------------

class TestLoadXCloud:
    def test_shape(self, manager):
        X_cloud = manager.load_X_cloud(identifier=42)
        assert X_cloud.shape == (NUM_TIMESTEPS, NUM_X_CLOUD_BANDS, H, W)

    def test_no_reshape(self, manager):
        X_cloud = manager.load_X_cloud(identifier=42, reshape=False)
        assert X_cloud.shape == (NUM_TIMESTEPS * NUM_X_CLOUD_BANDS, H, W)

    def test_return_dates(self, manager):
        X_cloud, dates = manager.load_X_cloud(identifier=42, return_dates=True)
        assert len(dates) == NUM_TIMESTEPS

    def test_return_as_dataset_raises(self, manager):
        with pytest.raises(NotImplementedError):
            manager.load_X_cloud(identifier=42, return_as_dataset=True)


# ---------------------------------------------------------------------------
# TestLoadY
# ---------------------------------------------------------------------------

class TestLoadY:
    def test_shape(self, manager):
        y = manager.load_y(identifier=42)
        assert y.shape == (H, W)

    def test_return_as_dataset_raises(self, manager):
        with pytest.raises(NotImplementedError):
            manager.load_y(identifier=42, return_as_dataset=True)

    def test_missing_identifier_raises(self, manager):
        with pytest.raises(KeyError):
            manager.load_y(identifier=999)


# ---------------------------------------------------------------------------
# TestLoadYId
# ---------------------------------------------------------------------------

class TestLoadYId:
    def test_shape(self, manager):
        y_id = manager.load_y_id(identifier=42)
        assert y_id.shape == (H, W)

    def test_return_as_dataset_raises(self, manager):
        with pytest.raises(NotImplementedError):
            manager.load_y_id(identifier=42, return_as_dataset=True)


# ---------------------------------------------------------------------------
# TestLoadRpgMask
# ---------------------------------------------------------------------------

class TestLoadRpgMask:
    def test_shape(self, manager):
        rpg = manager.load_rpg_mask(identifier=42, type="parcelles")
        assert rpg.shape == (H, W)

    def test_missing_rpg_key_raises(self):
        sample = make_sample()
        del sample["rpg_parcelles.tif"]
        with patch("hedgementation_utils.io.webdataset_io_manager.wds.WebDataset") as mock_wds:
            mock_wds.return_value = [sample]
            mgr = WDSHedgementationIOManager("/fake.tar", img_width=W, img_height=H)
        with pytest.raises(KeyError, match="rpg"):
            mgr.load_rpg_mask(identifier=42)

    def test_absent_rpg_mask_raises(self):
        sample = make_sample()
        del sample["rpg_parcelles.tif"]
        sample["rpg_other_type.tif"] = make_tif_bytes(1)
        with patch("hedgementation_utils.io.webdataset_io_manager.wds.WebDataset") as mock_wds:
            mock_wds.return_value = [sample]
            mgr = WDSHedgementationIOManager("/fake.tar", img_width=W, img_height=H)
        with pytest.raises(KeyError, match="rpg"):
            rpg = mgr.load_rpg_mask(identifier=42, type="parcelles")


# ---------------------------------------------------------------------------
# TestLoadAef
# ---------------------------------------------------------------------------

class TestLoadAef:
    def test_shape(self, manager):
        aef = manager.load_aef(identifier=42)
        assert aef.shape == (NUM_AEF_BANDS, H, W)

    def test_missing_identifier_raises(self, manager):
        with pytest.raises(KeyError):
            manager.load_aef(identifier=999)


# ---------------------------------------------------------------------------
# TestInheritedHighLevelMethods
# ---------------------------------------------------------------------------

class TestInheritedHighLevelMethods:
    def test_load_X_data_date_aligned_shapes(self, manager):
        X, X_cloud, dates = manager.load_X_data_date_aligned(identifier=42)
        assert X.shape[0] == X_cloud.shape[0] == len(dates)
        assert X.shape[1] == NUM_X_BANDS
        assert X_cloud.shape[1] == NUM_X_CLOUD_BANDS

    def test_load_X_data_date_aligned_consensus_dates(self, manager):
        X, X_cloud, dates = manager.load_X_data_date_aligned(identifier=42)
        assert set(dates) == {"2021-01-01", "2021-02-01"}

    def test_load_X_data_with_cloud_threshold_filters(self, manager):
        # First timestep has cloud score > threshold (all ones), second has low score (near zero).
        # Use a cloud score > threshold filter: scores > threshold pass.
        X_arr = np.zeros((NUM_TIMESTEPS, NUM_X_BANDS, H, W), dtype=np.float32)
        X_cloud_arr = np.zeros((NUM_TIMESTEPS, NUM_X_CLOUD_BANDS, H, W), dtype=np.float32)
        X_cloud_arr[0] = 1.0  # first timestep fully cloudy (high score = clear per Cloud Score+)
        X_cloud_arr[1] = 0.0  # second timestep low score (cloudy)
        dates = pd.Series(["2021-01-01", "2021-02-01"])

        manager.load_X_data_date_aligned = lambda identifier, **kw: (X_arr, X_cloud_arr, dates)
        X_out, _, dates_out = manager.load_X_data_with_cloud_threshold(
            identifier=42, cloud_threshold=0.5
        )
        assert len(dates_out) == 1
        assert list(dates_out)[0] == "2021-01-01"
