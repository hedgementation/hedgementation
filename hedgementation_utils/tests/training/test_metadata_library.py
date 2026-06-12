import pytest
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from unittest.mock import patch, MagicMock
from hedgementation_utils.training.metadata_library import MetadataLibrary, THZClass, SizeGroup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(n_per_fold=10, n_tilegroups=6):
    """Creates a synthetic metadata GeoDataFrame that mimics the real one."""
    rows = []
    thz_classes = [
        "TRC5: Subtropics, cool",
        "TRC4: Subtropics, moderately cool",
        "TRC7: Temperate, cool",
        "TRC6: Temperate, moderately cool",
    ]
    for fold in range(5):
        for i in range(n_per_fold):
            rows.append({
                "fold": fold,
                "tile_group": i % n_tilegroups,
                "thz_class": thz_classes[(fold * n_per_fold + i) % len(thz_classes)],
                "geometry": Point(fold, i)
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


@pytest.fixture
def library() -> MetadataLibrary:
    meta = make_metadata()
    lib = MetadataLibrary.__new__(MetadataLibrary)
    lib.metadata = meta
    lib.train_folds = [0, 1, 2]
    lib.valid_folds = [3]
    lib.test_folds = [4]
    lib.fold_dict = {
        "train": lib.train_folds,
        "valid": lib.valid_folds,
        "test": lib.test_folds,
    }
    return lib


# ---------------------------------------------------------------------------
# get_single_split
# ---------------------------------------------------------------------------

class TestGetSingleSplit:
    def test_returns_correct_folds(self, library):
        train = library.get_single_split(split="train")
        assert set(train["fold"].unique()) == {0, 1, 2}

    def test_valid_split(self, library):
        valid = library.get_single_split(split="valid")
        assert set(valid["fold"].unique()) == {3}

    def test_test_split(self, library):
        test = library.get_single_split(split="test")
        assert set(test["fold"].unique()) == {4}

    def test_invalid_split_raises(self, library):
        with pytest.raises(ValueError):
            library.get_single_split(split="nonexistent")

    def test_custom_metadata(self, library):
        subset = library.metadata[library.metadata["fold"] == 0]
        result = library.get_single_split(metadata=subset, split="train")
        assert set(result["fold"].unique()) == {0}


# ---------------------------------------------------------------------------
# get_split_metadata
# ---------------------------------------------------------------------------

class TestGetSplitMetadata:
    def test_returns_all_three_splits(self, library):
        splits = library.get_split_metadata()
        assert set(splits.keys()) == {"train", "valid", "test"}

    def test_splits_are_disjoint(self, library):
        splits = library.get_split_metadata()
        train_idx = set(splits["train"].index)
        valid_idx = set(splits["valid"].index)
        test_idx = set(splits["test"].index)
        assert train_idx.isdisjoint(valid_idx)
        assert train_idx.isdisjoint(test_idx)
        assert valid_idx.isdisjoint(test_idx)

    def test_string_query_applied_to_all_splits(self, library):
        splits = library.get_split_metadata(query='thz_class.str.contains("Temperate")')
        for split_df in splits.values():
            assert all("Temperate" in c for c in split_df["thz_class"])



# ---------------------------------------------------------------------------
# get_frames_for_THZ
# ---------------------------------------------------------------------------

class TestGetFramesForTHZ:
    def test_temperate_string(self, library):
        splits = library.get_frames_for_THZ(target="temperate", split=True)
        for df in splits.values():
            assert all("temperate" in c.lower() for c in df["thz_class"])

    def test_subtropic_string(self, library):
        splits = library.get_frames_for_THZ(target="subtropic", split=True)
        for df in splits.values():
            assert all("subtropic" in c.lower() for c in df["thz_class"])

    def test_thz_enum(self, library):
        splits = library.get_frames_for_THZ(target=THZClass.TEMPERATE, split=True)
        for df in splits.values():
            assert all("temperate" in c.lower() for c in df["thz_class"])

    def test_no_split_returns_geodataframe(self, library):
        result = library.get_frames_for_THZ(target="temperate", split=False)
        assert isinstance(result, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# get_geographic_distance_frames
# ---------------------------------------------------------------------------

class TestGetGeographicDistanceFrames:
    def test_excluded_group_absent_from_main_splits(self, library):
        for group_ind in range(6):
            splits = library.get_geographic_distance_frames(far_group_ind=group_ind)
            for split_name in ["train", "valid", "test"]:
                assert group_ind not in splits[split_name]["tile_group"].values, \
                    f"tile_group {group_ind} should be excluded from {split_name}"

    def test_test_far_contains_only_target_group(self, library):
        for group_ind in range(6):
            splits = library.get_geographic_distance_frames(far_group_ind=group_ind)
            assert set(splits["test_far"]["tile_group"].unique()) == {group_ind}

    def test_test_far_only_contains_test_folds(self, library):
        splits = library.get_geographic_distance_frames(far_group_ind=0)
        assert set(splits["test_far"]["fold"].unique()) == {4}

    def test_test_and_test_far_are_disjoint(self, library):
        splits = library.get_geographic_distance_frames(far_group_ind=0)
        assert set(splits["test"].index).isdisjoint(set(splits["test_far"].index))

    def test_test_and_test_far_cover_all_test_patches(self, library):
        """test + test_far should together equal the full test split."""
        for group_ind in range(6):
            splits = library.get_geographic_distance_frames(far_group_ind=group_ind)
            all_test = library.get_single_split(split="test")
            combined_idx = set(splits["test"].index) | set(splits["test_far"].index)
            assert combined_idx == set(all_test.index)

    def test_no_split_returns_geodataframe(self, library):
        result = library.get_geographic_distance_frames(far_group_ind=0, split=False)
        assert isinstance(result, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# downsample_to_other
# ---------------------------------------------------------------------------

class TestDownsampleToOther:
    def test_downsamples_when_left_larger(self, library):
        left = library.metadata.iloc[:20]
        right = library.metadata.iloc[:5]
        result = library.downsample_to_other(left, right)
        assert len(result) == 5

    def test_no_change_when_left_smaller(self, library):
        left = library.metadata.iloc[:3]
        right = library.metadata.iloc[:10]
        result = library.downsample_to_other(left, right)
        assert len(result) == 3

    def test_no_change_when_equal_size(self, library):
        left = library.metadata.iloc[:5]
        right = library.metadata.iloc[:5]
        result = library.downsample_to_other(left, right)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# downsample_all_splits
# ---------------------------------------------------------------------------

class TestDownsampleAllSplits:
    def test_downsamples_each_split(self, library):
        large_splits = library.get_split_metadata()
        small_meta = library.metadata.iloc[:6]  # 2 per fold
        small_splits = library.get_split_metadata(metadata=small_meta)
        result = library.downsample_all_splits(large_splits, small_splits)
        for k in result:
            assert len(result[k]) <= len(small_splits[k])

    def test_mismatched_keys_raises(self, library):
        splits = library.get_split_metadata()
        bad_splits = {"train": splits["train"]}
        with pytest.raises(ValueError):
            library.downsample_all_splits(splits, bad_splits)


# ---------------------------------------------------------------------------
# get_all_THZ_experiment_frames
# ---------------------------------------------------------------------------

class TestGetAllTHZExperimentFrames:
    def test_returns_expected_keys(self, library):
        frames = library.get_all_THZ_experiment_frames()
        assert set(frames.keys()) == {"temperate", "subtropic", "all_thz_classes"}

    def test_all_splits_present(self, library):
        frames = library.get_all_THZ_experiment_frames()
        for experiment in frames.values():
            assert set(experiment.keys()) == {"train", "valid", "test"}

    def test_all_experiments_same_size_per_split(self, library):
        frames = library.get_all_THZ_experiment_frames()
        for split in ["train", "valid"]:
            sizes = [len(frames[exp][split]) for exp in frames]
            assert len(set(sizes)) == 1, \
                f"Split '{split}' has unequal sizes across experiments: {sizes}"

    def test_temperate_only_contains_temperate(self, library):
        frames = library.get_all_THZ_experiment_frames()
        for split_df in frames["temperate"].values():
            assert all("temperate" in c.lower() for c in split_df["thz_class"])

    def test_subtropic_only_contains_subtropic(self, library):
        frames = library.get_all_THZ_experiment_frames()
        for split_df in frames["subtropic"].values():
            assert all("subtropic" in c.lower() for c in split_df["thz_class"])


# ---------------------------------------------------------------------------
# get_geographic_distance_experiment_frames
# ---------------------------------------------------------------------------

class TestGetGeographicDistanceExperimentFrames:
    def test_returns_one_experiment_per_tilegroup(self, library):
        frames = library.get_geographic_distance_experiment_frames(n_tilegroups=6)
        assert set(frames.keys()) == {f"far_group_{i}" for i in range(6)}

    def test_each_experiment_has_correct_splits(self, library):
        frames = library.get_geographic_distance_experiment_frames(n_tilegroups=6)
        for exp in frames.values():
            assert set(exp.keys()) == {"train", "valid", "test", "test_far"}


# ---------------------------------------------------------------------------
# SizeGroup filtering
# ---------------------------------------------------------------------------

def make_metadata_with_size_groups(n_per_group=8):
    """Synthetic metadata that includes a `size_group` column."""
    rows = []
    sizes = ["small", "medium", "large"]
    for fold in range(5):
        for i in range(n_per_group):
            rows.append({
                "fold": fold,
                "tile_group": i % 6,
                "thz_class": "TRC7: Temperate, cool",
                "size_group": sizes[i % 3],
                "geometry": Point(fold, i),
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestSizeGroup:
    def test_enum_ordering(self):
        assert SizeGroup.SMALL.value < SizeGroup.MEDIUM.value < SizeGroup.LARGE.value

    def test_coerce_from_string(self):
        assert SizeGroup.coerce("small") == SizeGroup.SMALL
        assert SizeGroup.coerce("LARGE") == SizeGroup.LARGE

    def test_coerce_passes_through_enum(self):
        assert SizeGroup.coerce(SizeGroup.MEDIUM) == SizeGroup.MEDIUM

    def test_small_keeps_only_small(self):
        meta = make_metadata_with_size_groups()
        filtered = MetadataLibrary._apply_size_group_filter(meta, SizeGroup.SMALL)
        assert set(filtered["size_group"].unique()) == {"small"}

    def test_medium_keeps_small_and_medium(self):
        meta = make_metadata_with_size_groups()
        filtered = MetadataLibrary._apply_size_group_filter(meta, SizeGroup.MEDIUM)
        assert set(filtered["size_group"].unique()) == {"small", "medium"}

    def test_large_keeps_all(self):
        meta = make_metadata_with_size_groups()
        filtered = MetadataLibrary._apply_size_group_filter(meta, SizeGroup.LARGE)
        assert set(filtered["size_group"].unique()) == {"small", "medium", "large"}
        assert len(filtered) == len(meta)

    def test_missing_column_returns_unchanged(self):
        meta = make_metadata()  # no size_group column
        filtered = MetadataLibrary._apply_size_group_filter(meta, SizeGroup.SMALL)
        assert len(filtered) == len(meta)
        assert "size_group" not in filtered.columns

    def test_init_does_not_filter_metadata(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        assert set(lib.metadata["size_group"].unique()) == {"small", "medium", "large"}
        assert len(lib.metadata) == len(meta)

    def test_single_instance_loads_each_size_group(self):
        """A single instance must be able to load small, medium, and large groups."""
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)

        small = lib.get_split_metadata(size_group=SizeGroup.SMALL)
        medium = lib.get_split_metadata(size_group=SizeGroup.MEDIUM)
        large = lib.get_split_metadata(size_group=SizeGroup.LARGE)

        for split in ["train", "valid", "test"]:
            assert set(small[split]["size_group"].unique()) == {"small"}
            assert set(medium[split]["size_group"].unique()) == {"small", "medium"}
            assert set(large[split]["size_group"].unique()) == {"small", "medium", "large"}

        # underlying metadata must remain untouched across loads
        assert set(lib.metadata["size_group"].unique()) == {"small", "medium", "large"}

    def test_no_size_group_returns_unfiltered(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        splits = lib.get_split_metadata()
        for split_df in splits.values():
            assert set(split_df["size_group"].unique()) == {"small", "medium", "large"}

    def test_size_group_string_arg(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        splits = lib.get_split_metadata(size_group="medium")
        for split_df in splits.values():
            assert set(split_df["size_group"].unique()) <= {"small", "medium"}

    def test_get_single_split_respects_size_group(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        train = lib.get_single_split(split="train", size_group=SizeGroup.SMALL)
        assert set(train["size_group"].unique()) == {"small"}

    def test_get_metadata_queried_respects_size_group(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        result = lib.get_metadata_queried(query="fold == 0", size_group=SizeGroup.MEDIUM)
        assert set(result["size_group"].unique()) <= {"small", "medium"}
        assert set(result["fold"].unique()) == {0}

    def test_explicit_metadata_overrides_size_group(self):
        """If `metadata=` is passed, it is used as-is; size_group is ignored."""
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        custom = meta[meta["size_group"] == "large"]
        splits = lib.get_split_metadata(metadata=custom, size_group=SizeGroup.SMALL)
        for split_df in splits.values():
            assert set(split_df["size_group"].unique()) <= {"large"}

    def test_size_group_propagates_to_thz_frames(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        splits = lib.get_frames_for_THZ(target="temperate", split=True, size_group=SizeGroup.SMALL)
        for split_df in splits.values():
            assert set(split_df["size_group"].unique()) <= {"small"}

    def test_size_group_propagates_to_geographic_frames(self):
        meta = make_metadata_with_size_groups()
        lib = MetadataLibrary(metadata=meta)
        splits = lib.get_geographic_distance_frames(far_group_ind=0, size_group=SizeGroup.MEDIUM)
        for split_df in splits.values():
            assert set(split_df["size_group"].unique()) <= {"small", "medium"}

    def test_missing_size_group_column_is_backwards_compatible(self):
        meta = make_metadata()  # no size_group column
        lib = MetadataLibrary(metadata=meta)
        splits = lib.get_split_metadata(size_group=SizeGroup.MEDIUM)
        total = sum(len(df) for df in splits.values())
        assert total == len(meta)