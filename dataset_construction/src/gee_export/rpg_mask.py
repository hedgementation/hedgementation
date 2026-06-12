import argparse
import os

import geopandas as gpd
import numpy as np
import shapely
from dotenv import load_dotenv
from rasterio.features import rasterize
from shapely.affinity import affine_transform

load_dotenv()

DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
RPG_MASK_SUBDIR = os.environ.get("RPG_MASK_SUBDIR", "rpg_masks")
MASK_SAVE_DIR = os.path.join(DATASET_ROOT, RPG_MASK_SUBDIR)

RPG_PATH = os.getenv(
    "RPG_PATH",
    "/scratch/nathan/data/RPG_2-0_GPKG_LAMB93_FXX-2022/PARCELLES_GRAPHIQUES.gpkg"
)


def rescale_geom(multipoly, x_length=128, y_length=128):
    minx, miny, maxx, maxy = multipoly.bounds

    width = maxx - minx
    height = maxy - miny

    scale_x = x_length / width
    scale_y = y_length / height

    affine_matrix = [
        scale_x,
        0,
        0,
        -scale_y,
        -minx * scale_x,
        maxy * scale_y
    ]

    return affine_transform(multipoly, affine_matrix)


def rasterize_rpg_geom(geom,
                       bounding_box,
                       height=128,
                       width=128):
    geom = shapely.intersection(
        shapely.make_valid(geom),
        shapely.make_valid(bounding_box)
    )

    if geom.is_empty:
        return np.zeros((height, width), dtype=np.uint8)

    geom = rescale_geom(geom, height, width)

    return rasterize(
        [geom],
        out_shape=(height, width),
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True,
    )


def compute_rpg_mask_for_patch(patch_geom,
                                rpg_gdf: gpd.GeoDataFrame,
                                patch_crs: str = "EPSG:3857",
                                height: int = 128,
                                width: int = 128,
                                buffer_width: int = None) -> tuple[np.ndarray, float]:
    """Compute the RPG agricultural mask and coverage fraction for a single patch.

    Args:
        patch_geom: Patch bounding-box geometry as a Shapely object.
        rpg_gdf: RPG GeoDataFrame (pre-loaded; any CRS is accepted).
        patch_crs: CRS of ``patch_geom`` (e.g. ``"EPSG:3857"`` or ``"EPSG:4326"``).
        height, width: Output raster dimensions in pixels.
        buffer_width: Optional buffer in metres to apply to RPG parcel geometries
            before rasterization (requires rpg_gdf to be in a metric CRS).

    Returns:
        ``(mask, rpg_coverage)`` where ``mask`` is a uint8 ndarray of shape
        ``(height, width)`` and ``rpg_coverage`` is the fraction of the patch
        area covered by agricultural parcels.
    """
    # Reproject patch geometry to match RPG CRS
    patch_series = gpd.GeoSeries([patch_geom], crs=patch_crs).to_crs(rpg_gdf.crs)
    patch_in_rpg_crs = patch_series.iloc[0]

    candidates = rpg_gdf[rpg_gdf.intersects(patch_in_rpg_crs)]
    if candidates.empty:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    geoms = candidates.geometry.values
    if buffer_width is not None:
        geoms = shapely.buffer(geoms, buffer_width)
    rpg_union = shapely.unary_union(geoms)
    rpg_coverage = float(
        shapely.area(shapely.intersection(shapely.make_valid(patch_in_rpg_crs), shapely.make_valid(rpg_union)))
        / shapely.area(patch_in_rpg_crs)
    )

    mask = rasterize_rpg_geom(rpg_union, patch_in_rpg_crs, height, width)
    return mask, rpg_coverage


def merge_rpg_and_bda(rpg: gpd.GeoDataFrame,
                      bda: gpd.GeoDataFrame,
                      final_crs: str = "EPSG:4326",
                      id_col: str = "identifier",
                      join: str = "left",
                      add_masks: bool = True,
                      buffer_width=None,
                      mask_dir: str = MASK_SAVE_DIR):
    """Join RPG parcel data onto BDA patch metadata and optionally compute masks.

    Note: pixel-level metadata statistics (masked pixel counts, hedgerow overlap,
    etc.) are computed in a separate post-processing step. See
    ``src/postprocessing/rpg_metadata.py``.
    """
    from src.utils.io_utils import save_rpg_mask

    if buffer_width is not None:
        rpg = rpg.copy()
        rpg.geometry = rpg.geometry.apply(lambda x: x.buffer(buffer_width))

    rpg_crs = rpg.crs
    rpg["rpg_geometry"] = rpg.geometry
    joined = bda.to_crs(rpg_crs).sjoin(rpg, how=join).to_crs(final_crs)

    bda_cols = [col for col in bda.columns if col not in ['geometry', id_col]]
    rpg_cols = [col for col in rpg.columns if col != 'geometry']

    agg_dict = {col: 'first' for col in bda_cols}
    agg_dict['geometry'] = 'first'
    agg_dict.update({col: lambda x: list(x) for col in rpg_cols})
    agg_dict['rpg_geometry'] = lambda x: shapely.unary_union(list(x))

    if 'index_right' in joined.columns:
        agg_dict['index_right'] = lambda x: list(x)

    merged_dataset: gpd.GeoDataFrame = (
        joined
        .groupby(by=id_col)
        .agg(agg_dict)
        .reset_index()
    )
    merged_dataset["rpg_geometry"] = gpd.GeoSeries(
        merged_dataset["rpg_geometry"], crs=rpg_crs
    ).to_crs(final_crs)

    merged_dataset["rpg_coverage"] = merged_dataset.apply(
        lambda x: shapely.area(
            shapely.intersection(x.geometry, shapely.make_valid(x["rpg_geometry"]))
        ) / shapely.area(x.geometry),
        axis="columns"
    )

    if add_masks:
        merged_dataset["rpg_mask"] = merged_dataset.apply(
            lambda x: rasterize_rpg_geom(x["rpg_geometry"], x["geometry"]),
            axis=1
        )
        merged_dataset.apply(
            lambda x: save_rpg_mask(mask=x["rpg_mask"], identifier=x["ID_PATCH"], mask_dir=mask_dir),
            axis=1
        )
        merged_dataset = merged_dataset.drop(columns=["rpg_mask"])

    return merged_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpg_path", default=RPG_PATH)
    parser.add_argument("--metadata_path", default=os.path.join(DATASET_ROOT, "metadata.geojson"))
    parser.add_argument("--save_path", default=os.path.join(DATASET_ROOT, "metadata_with_rpg.geojson"))
    parser.add_argument("--add_masks", action="store_true")
    parser.add_argument("--buffer_width", default=None, type=int)

    args = parser.parse_args()

    rpg = gpd.read_file(args.rpg_path, use_arrow=True, engine='pyogrio').to_crs("EPSG:3857")
    bda = gpd.read_file(args.metadata_path, use_arrow=True, engine='pyogrio').to_crs("EPSG:3857")
    bda["identifier"] = bda.index

    rpg_and_bda = gpd.GeoDataFrame(
        merge_rpg_and_bda(rpg, bda, add_masks=args.add_masks, buffer_width=args.buffer_width)
    )
    rpg_and_bda["rpg_geometry"] = rpg_and_bda["rpg_geometry"].to_wkt()
    rpg_and_bda.to_file(args.save_path)
