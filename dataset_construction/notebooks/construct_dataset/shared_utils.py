import os
import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shapely
import torch

try:
    RANDOM_SEED = int(os.environ["RANDOM_SEED"])
except:
    RANDOM_SEED = 15


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

parent_rng = np.random.default_rng(seed=RANDOM_SEED)



def get_new_random_seed():
    return parent_rng.integers(0,2**32)

def sample_gdf(gdf, folds, split, n_per_fold=600):
    dfs = []

    value_pcts = gdf["tile"].value_counts().apply(lambda x: x / len(gdf))
    for f in folds:
        inner_gdfs = []
        for t in pd.unique(gdf["tile"]):
            sub_gdf = gdf[(gdf["tile"] == t) & (gdf["fold"] == f)]
            if len(sub_gdf) == 0:
                continue
            pct = float(value_pcts.loc[t]) 
            child_rng = np.random.default_rng(get_new_random_seed())
            n_t = n_per_fold * pct

            inds = child_rng.choice([i for i in range(len(sub_gdf))], int(round(n_t)))
            sub_gdf_sampled = sub_gdf.iloc[inds]
            inner_gdfs.append(sub_gdf_sampled)
        
        full_inner_gdf = pd.concat(inner_gdfs, axis=0)
        dfs.append(full_inner_gdf)
    full_gdf = pd.concat(dfs, axis=0)
    full_gdf["split"] = [split] * len(full_gdf)
    return full_gdf


def assert_geodataframes_equal(gdf1, gdf2, geometry_columns=None,geom_only=False,geom_tol=1e-8):

    if geometry_columns is None:
        geometry_columns = ["geometry", "hedgerows"]
    cols = [c for c in sorted(gdf1.columns) if c not in geometry_columns]
    if not geom_only:
        pd.testing.assert_frame_equal(
            gdf1[cols].reset_index(drop=True),
            gdf2[cols].reset_index(drop=True),
            check_dtype=False
        )
    for col in geometry_columns:
        for i, (g1, g2) in enumerate(zip(gdf1[col], gdf2[col])):
            if isinstance(g1,str):
                g1 = shapely.wkt.loads(g1)
            if isinstance(g2,str):
                g2 = shapely.wkt.loads(g2)
            # Compare canonical forms: file writers disagree on ring start /
            # winding and multipart order, and reprojection adds float drift;
            # neither is a real geometric difference, but equals_exact on the
            # raw geometries is sensitive to both.
            g1n = shapely.normalize(g1)
            g2n = shapely.normalize(g2)
            if not g1n.equals_exact(g2n, tolerance=geom_tol):
                hd = shapely.hausdorff_distance(g1n, g2n)
                raise AssertionError(
                    f"Row {i}, column {col}: geometries differ "
                    f"(hausdorff distance {hd:.2e}, tolerance {geom_tol:.0e})\n"
                    f"  g1: {g1}\n  g2: {g2}"
                )
                

# def sample_gdf(gdf, folds, split, random_seed_func, n_per_fold=600):
#     dfs = []

#     value_pcts = gdf["tile"].value_counts().apply(lambda x: x / len(gdf))
#     for f in folds:
#         inner_gdfs = []
#         for t in pd.unique(gdf["tile"]):
#             sub_gdf = gdf[(gdf["tile"] == t) & (gdf["fold"] == f)]
#             if len(sub_gdf) == 0:
#                 continue
#             pct = float(value_pcts.loc[t]) 
#             child_rng = np.random.default_rng(random_seed_func())
#             n_t = n_per_fold * pct

#             inds = child_rng.choice([i for i in range(len(sub_gdf))], int(round(n_t)))
#             sub_gdf_sampled = sub_gdf.iloc[inds]
#             inner_gdfs.append(sub_gdf_sampled)
        
#         full_inner_gdf = pd.concat(inner_gdfs, axis=0)
#         dfs.append(full_inner_gdf)
#     full_gdf = pd.concat(dfs, axis=0)
#     full_gdf["split"] = [split] * len(full_gdf)
#     return full_gdf

def plot_aez_counts(gdf,colors,title="AEZ Counts"):
        value_counts = gdf["aez_class"].value_counts().sort_values(ascending=False)
        fig = plt.figure()
        fig.set_figwidth(10)
        bars = plt.bar(range(len(value_counts)), value_counts, color=colors, edgecolor='black', linewidth=0.5)

        plt.xlabel('AEZ class', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.xticks(range(len(value_counts)))
        plt.title(title)
        legend_labels = [f"{i}:{label[:50]}{'...' if len(label) > 50 else ''}" 
                        for i, label in enumerate(value_counts.index)]

        plt.legend(bars, legend_labels, 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                framealpha=0.9,
                title='Class Labels',
                title_fontsize=11)

        plt.tight_layout()
        plt.show()
