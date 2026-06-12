"""
Dataclass configs for the GEE export pipeline, backed by JSON files under
configs/gee_export/. Covers the shared date range / patch grid used by all
satellite time-series exports, per-collection settings (Sentinel-2, cloud
score, AEF embeddings), and the downsampling procedure used to generate the
y / y_id rasters. Each config can be re-pointed to a different JSON file via
CLI arguments on the export scripts.
"""
import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "gee_export"


def normalize_date(date_str: str) -> str:
    """Normalize 'YYYY-MM-DD' or legacy 'YYYYMMDD' to 'YYYY-MM-DD'."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {date_str!r} (expected YYYY-MM-DD or YYYYMMDD)")


@dataclass
class SharedExportConfig:
    """Date range and patch-grid parameters shared by all satellite time-series exports."""
    start_date: str
    end_date: str
    crs: str = "EPSG:3857"
    scale: int = 10  # meters/pixel of the exported rasters
    nb_pixel: int = 128  # exported images are nb_pixel x nb_pixel

    def __post_init__(self):
        self.start_date = normalize_date(self.start_date)
        self.end_date = normalize_date(self.end_date)


@dataclass
class Sentinel2Config:
    """Sentinel-2 time-series export (X). Dates come from SharedExportConfig."""
    collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    selected_bands: List[str] = field(default_factory=lambda: [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
    ])
    cloud_threshold: int = 20  # max CLOUDY_PIXEL_PERCENTAGE when cloud filtering


@dataclass
class CloudScoreConfig:
    """Cloud score mask export (X_cloud). Dates default to the shared range."""
    collection: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    selected_bands: List[str] = field(default_factory=lambda: ["cs", "cs_cdf"])
    start_date: Optional[str] = None  # None -> inherit SharedExportConfig
    end_date: Optional[str] = None


@dataclass
class AEFConfig:
    """AEF satellite embedding export (X_aef).

    `windows` is a list of [start, end] date pairs; the annual embeddings for
    each window are averaged. None -> a single window spanning the shared range.
    """
    collection: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    windows: Optional[List[List[str]]] = None


@dataclass
class DownsamplingConfig:
    """Rasterization/downsampling procedure for the y and y_id rasters.

    Hedgerow linestrings are buffered to `width` meters, rasterized at
    `initial_scale` m/pixel, then (if `reduce_resolution`) reduced to
    `final_scale` m/pixel (mean for y, mode for y_id). final_scale=None
    inherits SharedExportConfig.scale.
    """
    width: float = 7
    initial_scale: float = 0.5
    reduce_resolution: bool = True
    final_scale: Optional[float] = None


@dataclass
class ExportConfigs:
    """Bundle of all export configs with inheritance from `shared` resolved."""
    shared: SharedExportConfig
    satellite: Sentinel2Config
    cloud_score: CloudScoreConfig
    aef: AEFConfig
    downsampling: DownsamplingConfig


def _load_dataclass_json(path: Optional[str], default_name: str, cls):
    """Load a dataclass from a JSON file, rejecting unknown keys."""
    resolved = Path(path) if path else DEFAULT_CONFIG_DIR / default_name
    with open(resolved) as f:
        data = json.load(f)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in {resolved}: {sorted(unknown)} "
            f"(expected a subset of {sorted(known)})"
        )
    return cls(**data)


def load_export_configs(shared_path: Optional[str] = None,
                        satellite_path: Optional[str] = None,
                        cloud_score_path: Optional[str] = None,
                        aef_path: Optional[str] = None,
                        downsampling_path: Optional[str] = None) -> ExportConfigs:
    """Load all export configs, falling back to the defaults in configs/gee_export/.

    Inheritance from the shared config (cloud score dates, AEF windows,
    downsampling final_scale) is resolved here, so consumers can use every
    field directly.
    """
    shared = _load_dataclass_json(shared_path, "shared_export.json", SharedExportConfig)
    satellite = _load_dataclass_json(satellite_path, "sentinel2.json", Sentinel2Config)
    cloud_score = _load_dataclass_json(cloud_score_path, "cloud_score.json", CloudScoreConfig)
    aef = _load_dataclass_json(aef_path, "aef.json", AEFConfig)
    downsampling = _load_dataclass_json(downsampling_path, "downsampling.json", DownsamplingConfig)

    cloud_score.start_date = (normalize_date(cloud_score.start_date)
                              if cloud_score.start_date else shared.start_date)
    cloud_score.end_date = (normalize_date(cloud_score.end_date)
                            if cloud_score.end_date else shared.end_date)

    if aef.windows is None:
        aef.windows = [[shared.start_date, shared.end_date]]
    else:
        aef.windows = [[normalize_date(s), normalize_date(e)] for s, e in aef.windows]

    if downsampling.final_scale is None:
        downsampling.final_scale = shared.scale

    return ExportConfigs(shared=shared, satellite=satellite, cloud_score=cloud_score,
                         aef=aef, downsampling=downsampling)
