import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import geopandas as gpd
import folium

from src.training.train_utils import ModelFactory, Backbones
from src.training.dataset_dataloader import setup_dataloader
from src.training.experiments_registry import deserialize_label_fn


class DensityMapGenerator:
    def __init__(self, model_dir, dataset_root):
        self.model_dir = model_dir
        self.dataset_root = dataset_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(self.model_dir, "params.json"), "r") as f:
            self.params = json.load(f)

        self.model = self._load_model()
        self.results = None   # {split_name: tensor} dans l'ordre du loader
        self.loaders = {}     # {split_name: DataLoader}

    def _load_model(self):
        additional_model_params = self.params.get("additional_model_params") or {}
        additional_model_params["out_conv_last_relu"] = False
        checkpoint_path = os.path.join(self.model_dir, "checkpoints", "best_model_params.pt")
        model = ModelFactory.construct_model(
            backbone=Backbones[self.params["backbone"].upper()],
            num_buckets=int(self.params["num_buckets"]),
            additional_model_params=additional_model_params,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Inference: binary nuisance (p0 / f0)
    # ------------------------------------------------------------------

    def generate_all_data(self, split_dict, save=True, is_regression=False):
        """
        Run inference for binary / regression tasks.

        Stores results[split_name]:
          - is_regression=False : (N, 2, H, W)  — channel 0: p0, channel 1: f0
          - is_regression=True  : (N, C, H, W)  — raw model output
        """
        self.results = {}

        for split_name, df in split_dict.items():
            print(f"\n--- Inference : {split_name} ({len(df)} patchs) ---")

            transfer = deserialize_label_fn(self.params["transfer"])

            self.loaders[split_name] = setup_dataloader(
                metadata_frame=df,
                data_path=self.dataset_root,
                num_buckets=self.params["num_buckets"],
                image_count=self.params["image_count"],
                normalization=self.params["normalization"].upper(),
                y_transform=self.params["y_transform"].upper(),
                inclusion_intervals=self.params["inclusion_intervals"],
                batch_size=self.params["batch_size"],
                shuffle=False,
                transfer=transfer,
                load_X_cloud=True,
                cloud_threshold=self.params["cloud_threshold"],
                cloud_band=self.params["cloud_band"], 
            )

            all_batches = []
            with torch.no_grad():
                for batch in tqdm(self.loaders[split_name]):
                    inputs, *_ = batch
                    inputs, dates = inputs[0], inputs[1]
                    outputs = self.model(
                        inputs.to(self.device),
                        batch_positions=dates.to(self.device),
                    )

                    if is_regression:
                        all_batches.append(outputs.cpu())
                    else:
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        f0 = probs / (1 - probs + 1e-12)
                        all_batches.append(torch.stack([probs, f0], dim=1).cpu())

            self.results[split_name] = torch.cat(all_batches, dim=0)

        if save:
            torch.save(self.results, os.path.join(self.model_dir, "all_density_results.pt"))

        return self.results

    # ------------------------------------------------------------------
    # Inference: multi-class tile number
    # ------------------------------------------------------------------

    def generate_tile_data(self, split_dict, save=False):
        """
        Run inference for multi-class tile-number classification.

        Stores results[split_name]: (N, H, W) int64 — argmax predicted tile
        index per pixel.

        Also stores for each split:
          - majority_vote[split_name] : (N,) int64 — majority tile per patch
          - entropy[split_name]       : (N,) float — pixel-distribution entropy
        """
        self.results     = {}
        self.majority_vote = {}
        self.entropy       = {}

        num_classes = int(self.params["num_buckets"]) + 1

        for split_name, df in split_dict.items():
            print(f"\n--- Tile inference : {split_name} ({len(df)} patchs) ---")

            transfer = deserialize_label_fn(self.params.get("transfer"))

            self.loaders[split_name] = setup_dataloader(
                metadata_frame=df,
                data_path=self.dataset_root,
                num_buckets=int(self.params["num_buckets"]),
                image_count=int(self.params["image_count"]),
                normalization=self.params["normalization"].upper(),
                y_transform=self.params["y_transform"].upper(),
                inclusion_intervals=self.params.get("inclusion_intervals"),
                batch_size=int(self.params["batch_size"]),
                shuffle=False,
                transfer=transfer,
                load_X_cloud=True,
                cloud_threshold=self.params["cloud_threshold"],
                cloud_band=self.params["cloud_band"], 
            )

            pred_maps_split = []
            label_list = []
            majority_list = []
            entropy_list = []

            with torch.no_grad():
                for batch in tqdm(self.loaders[split_name]):
                    input_tuple, y_batch, *_ = batch
                    inputs = input_tuple[0].to(self.device)
                    dates = input_tuple[1].to(self.device)

                    outputs  = self.model(inputs, batch_positions=dates)  # (B, C, H, W)
                    pred_cls = torch.argmax(outputs, dim=1).cpu()          # (B, H, W)

                    B = pred_cls.shape[0]
                    for i in range(B):
                        pm = pred_cls[i].numpy()
                        pred_maps_split.append(pm)
                        majority_list.append(int(np.bincount(pm.flatten(), minlength=num_classes).argmax()))
                        label_list.append(int(y_batch[i, 0, 0].item()))
                        entropy_list.append(self._pixel_entropy(pm, num_classes))

            self.results[split_name] = np.stack(pred_maps_split)          # (N, H, W)
            self.majority_vote[split_name] = np.array(majority_list)
            self.labels = np.array(label_list)   # shared across splits (same order)
            self.entropy[split_name] = np.array(entropy_list)

        if save:
            torch.save(
                {k: torch.from_numpy(v) for k, v in self.results.items()},
                os.path.join(self.model_dir, "tile_inference_results.pt"),
            )

        return self.results

    @staticmethod
    def _pixel_entropy(pred_map: np.ndarray, num_classes: int) -> float:
        """Shannon entropy of the pixel-class histogram for one patch."""
        counts = np.bincount(pred_map.flatten(), minlength=num_classes).astype(float)
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    # ------------------------------------------------------------------
    # Visualisation: tile pixel mask (left: RGB, right: coloured mask)
    # ------------------------------------------------------------------

    def display_tile_mask(self, split_name: str, patch_idx: int,
                          class_to_color: dict | None = None,
                          tile_remap_inv: dict = {},
                          subdivisions: int | None = 1,
                          save_path: str | None = None):
        """
        Two-panel figure:
          Left  — median RGB image of the patch.
          Right — per-pixel tile prediction mask, one colour per tile class,
                  with a legend.

        Parameters
        ----------
        split_name    : key in self.results / self.loaders
        patch_idx     : index in the split's metadata_frame
        class_to_color: optional dict {tile_idx: (r,g,b,a)}.
                        Built from tab20 automatically if not provided.
        save_path     : optional path to save the figure
        """
        pred_map = self.results[split_name][patch_idx]        # (H, W) int
        loader   = self.loaders[split_name]
        patch_id = loader.dataset.metadata_frame.iloc[patch_idx]["ID_PATCH"]

        # RGB image
        sample      = loader.dataset[patch_idx]
        input_tuple = sample[0]
        X           = input_tuple[0]                          # (T, C, H, W)
        img_rgb     = X.median(dim=0)[0][[2, 1, 0]].permute(1, 2, 0).numpy()
        img_rgb     = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        img_rgb     = np.clip(img_rgb, 0, 1)

        # Colour map
        present_classes = sorted(np.unique(pred_map).tolist())
        if class_to_color is None:
            cmap_tab       = plt.get_cmap("tab20", max(len(present_classes), 1))
            class_to_color = {cls: cmap_tab(i) for i, cls in enumerate(present_classes)}

        H, W = pred_map.shape
        mask_rgb = np.zeros((H, W, 3))
        for cls in present_classes:
            r, g, b, *_ = class_to_color[cls]
            mask_rgb[pred_map == cls] = [r, g, b]

        majority   = self.majority_vote[split_name][patch_idx]
        true_tile  = self.labels[patch_idx] if hasattr(self, "labels") else "?"
        entropy    = self.entropy[split_name][patch_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(img_rgb)
        axes[0].set_title(f"RGB — ID_PATCH {patch_id}\nTrue tile: {true_tile}  |  Pred: {majority}")
        axes[0].axis("off")

        axes[1].imshow(mask_rgb, interpolation="nearest")
        axes[1].set_title(f"Pixel-level tile predictions  |  Entropy: {entropy:.3f}")
        axes[1].axis("off")

        if subdivisions != 1 and tile_remap_inv != {}:
            tile_orig = tile_remap_inv[cls // (subdivisions ** 2)]
            subtile = cls % (subdivisions ** 2)
        else:
            tile_orig = cls 
            subtile = ""

        legend_handles = [
            Patch(facecolor=class_to_color[cls], label=f"tile {tile_orig} / sub {subtile}")
            for cls in present_classes
        ]
        axes[1].legend(
            handles=legend_handles,
            loc="upper right", bbox_to_anchor=(1.30, 1.0),
            fontsize=8, title="Predicted tile",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"Saved: {save_path}")
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Visualisation: patch location on France map
    # ------------------------------------------------------------------

    def display_patch_on_france_map(
            self,
            split_name: str,
            patch_idx: int,
            padding: float = 0.2,
            title: str | None = None,
            tile_remap_inv: dict | None = None,
            subdivisions: int | None = 1,
            save_path: str | None = None,
            show_tile_boundaries: bool = True,
        ):

        import os
        import numpy as np
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from shapely.geometry import box

        # --------------------------------------------------
        # LOAD
        # --------------------------------------------------
        loader = self.loaders[split_name]

        metadata = gpd.read_file(
            os.path.join(self.dataset_root, "metadata.geojson")
        )

        france = self._load_france_boundaries()

        meta_df = loader.dataset.metadata_frame.reset_index(drop=True)

        # --------------------------------------------------
        # TARGET PATCH
        # --------------------------------------------------
        patch_id = meta_df.iloc[patch_idx]["ID_PATCH"]

        target_row = metadata[metadata["ID_PATCH"] == patch_id]

        if target_row.empty:
            print(f"Patch {patch_id} not found.")
            return

        target_geom = target_row.iloc[0].geometry

        minx, miny, maxx, maxy = target_geom.bounds

        # --------------------------------------------------
        # VIEW WINDOW
        # --------------------------------------------------
        dx = max(maxx - minx, 1e-4)
        dy = max(maxy - miny, 1e-4)

        view_box = box(
            minx - padding * dx,
            miny - padding * dy,
            maxx + padding * dx,
            maxy + padding * dy,
        )

        # --------------------------------------------------
        # LOCAL CONTEXT ONLY  <-- IMPORTANT FIX
        # --------------------------------------------------
        local_metadata = metadata[metadata.intersects(view_box)].copy()

        # --------------------------------------------------
        # LABELS
        # --------------------------------------------------
        majority = self.majority_vote[split_name][patch_idx]

        true_tile = (
            self.labels[patch_idx]
            if hasattr(self, "labels")
            else majority
        )

        is_correct = (majority == true_tile)

        # --------------------------------------------------
        # FIGURE
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))

        # France background
        france.plot(
            ax=ax,
            color="lightgrey",
            edgecolor="black",
            linewidth=0.5,
            zorder=1,
        )

        # --------------------------------------------------
        # LOCAL PATCHES ONLY
        # --------------------------------------------------
        for _, row in local_metadata.iterrows():

            geom = row.geometry
            x0, y0, x1, y1 = geom.bounds

            is_target = (row["ID_PATCH"] == patch_id)

            ax.add_patch(Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.5 if is_target else 0.3,
                edgecolor="black" if is_target else "grey",
                facecolor=(
                    "green"
                    if (is_target and is_correct)
                    else "red"
                    if (is_target and not is_correct)
                    else "grey"
                ),
                alpha=0.7 if is_target else 0.25,
                zorder=5 if is_target else 3,
            ))

        # --------------------------------------------------
        # TARGET LABEL
        # --------------------------------------------------
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        ax.text(
            cx,
            maxy,
            f"ID {patch_id}",
            fontsize=8,
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
            zorder=6,
        )

        # --------------------------------------------------
        # TILE BOUNDARIES (LOCAL ONLY)
        # --------------------------------------------------
        if show_tile_boundaries:

            local_tiles = (
                local_metadata.groupby("tile")["geometry"]
                .agg(lambda g: g.unary_union.envelope)
            )

            for tile_id, tile_geom in local_tiles.items():

                tx0, ty0, tx1, ty1 = tile_geom.bounds

                ax.add_patch(Rectangle(
                    (tx0, ty0),
                    tx1 - tx0,
                    ty1 - ty0,
                    linewidth=1.0,
                    edgecolor="steelblue",
                    facecolor="none",
                    zorder=4,
                ))

                ax.text(
                    (tx0 + tx1) / 2,
                    (ty0 + ty1) / 2,
                    str(tile_id),
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="steelblue",
                    fontweight="bold",
                    zorder=6,
                )

        # --------------------------------------------------
        # ZOOM
        # --------------------------------------------------
        ax.set_xlim(view_box.bounds[0], view_box.bounds[2])
        ax.set_ylim(view_box.bounds[1], view_box.bounds[3])

        ax.set_aspect("equal")

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------
        if subdivisions != 1 and tile_remap_inv != {}:
            true_orig  = tile_remap_inv[true_tile  // (subdivisions ** 2)]
            pred_orig  = tile_remap_inv[majority   // (subdivisions ** 2)]
            true_sub   = true_tile  % (subdivisions ** 2)
            pred_sub   = majority   % (subdivisions ** 2)

        else:
            true_orig = true_tile 
            pred_orig = majority 
            true_sub = ""
            pred_sub = ""
        
        ax.set_title(
            title or (
                f"Patch {patch_id} — "
                f"True: {true_orig}/{true_sub} | "
                f"Pred: {pred_orig}/{pred_sub} | "
                f"{'✓ correct' if is_correct else '✗ wrong'}"
            )
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------
        if save_path:
            plt.savefig(save_path, dpi=80)

        plt.show()
        plt.close(fig)
    # ------------------------------------------------------------------
    # Visualisation: all patches on France map (green/red by correctness)
    # ------------------------------------------------------------------

    def display_tile_predictions_on_map(
            self,
            split_name: str,
            title: str | None = None,
            save_path: str | None = None,
            tile_remap_inv: dict | None = None,
            subdivisions: int | None = 0,
            focus_tile: int | None = None
        ):
        """
        Map of France with one rectangle per patch coloured green (correct)
        or red (wrong). Tile boundaries drawn in blue with tile index label.

        If focus_tile is provided, zoom only on that tile and display only its patches.
        """

        import numpy as np
        import geopandas as gpd
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        loader   = self.loaders[split_name]
        metadata = gpd.read_file(os.path.join(self.dataset_root, "metadata.geojson"))
        meta_df_full = loader.dataset.metadata_frame.reset_index(drop=True)

        majority_arr = self.majority_vote[split_name]
        labels_arr   = self.labels if hasattr(self, "labels") else majority_arr

        # -----------------------------
        # TILE FILTERING (CLEAN + SAFE)
        # -----------------------------
        if focus_tile is not None:
            metadata_tile = metadata[metadata["tile"] == focus_tile]

            if metadata_tile.empty:
                raise ValueError(f"Tile {focus_tile} not found in metadata")

            tile_patch_ids = metadata_tile["ID_PATCH"].values
            mask = meta_df_full["ID_PATCH"].isin(tile_patch_ids).values

            meta_df = meta_df_full[mask].reset_index(drop=True)
            majority_arr = np.array(majority_arr)[mask]
            labels_arr   = np.array(labels_arr)[mask]

            france = None
        else:
            meta_df = meta_df_full
            metadata_tile = metadata
            france = self._load_france_boundaries()

        # -----------------------------
        # FIGURE
        # -----------------------------
        fig, ax = plt.subplots(figsize=(14, 14))

        if france is not None:
            france.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5)

        correct_count = 0

        # -----------------------------
        # PATCHES
        # -----------------------------
        for i in range(len(meta_df)):
            patch_id = meta_df.iloc[i]["ID_PATCH"]

            row = metadata[metadata["ID_PATCH"] == patch_id]
            if row.empty:
                continue

            minx, miny, maxx, maxy = row.iloc[0].geometry.bounds

            is_correct = (majority_arr[i] == labels_arr[i])
            correct_count += int(is_correct)

            ax.add_patch(plt.Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                linewidth=0.5,
                edgecolor="black",
                facecolor="green" if is_correct else "red",
                alpha=0.55,
                zorder=4,
            ))
            if subdivisions != 1 and tile_remap_inv != {}:
                tile_orig = tile_remap_inv[majority_arr[i] // (subdivisions ** 2)],
                subtile = majority_arr[i] % (subdivisions ** 2)
            else:
                tile_orig = majority_arr[i]
                subtile = ""
            ax.text(
                (minx + maxx) / 2,
                (miny + maxy) / 2,
                f"{tile_orig}/{subtile}",
                fontsize=6 if focus_tile is not None else 4,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                zorder=5,
            )

        # -----------------------------
        # TILE BOUNDARIES (global only)
        # -----------------------------
        if focus_tile is None:
            tile_boundaries = (
                metadata.groupby("tile")["geometry"]
                .agg(lambda geoms: geoms.unary_union.envelope)
            )

            for tile_id, tile_geom in tile_boundaries.items():
                minx, miny, maxx, maxy = tile_geom.bounds

                ax.add_patch(plt.Rectangle(
                    (minx, miny),
                    maxx - minx,
                    maxy - miny,
                    linewidth=1.5,
                    edgecolor="steelblue",
                    facecolor="none",
                    zorder=3,
                ))

                ax.text(
                    (minx + maxx) / 2,
                    (miny + maxy) / 2,
                    str(tile_id),
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="steelblue",
                    fontweight="bold",
                    zorder=6,
                )

        # -----------------------------
        # ZOOM LOGIC
        # -----------------------------
        if focus_tile is not None:
            bounds = metadata_tile.total_bounds  # (minx, miny, maxx, maxy)

            minx, miny, maxx, maxy = bounds

            if not np.all(np.isfinite(bounds)):
                raise ValueError(f"Invalid bounds for tile {focus_tile}: {bounds}")

            padding_x = (maxx - minx) * 0.05
            padding_y = (maxy - miny) * 0.05

            ax.set_xlim(minx - padding_x, maxx + padding_x)
            ax.set_ylim(miny - padding_y, maxy + padding_y)

        else:
            ax.set_xlim(-5.5, 9.5)
            ax.set_ylim(41.0, 51.5)

        # -----------------------------
        # LEGEND
        # -----------------------------
        ax.legend(handles=[
            Patch(facecolor="green", edgecolor="black", alpha=0.6, label="Correct tile"),
            Patch(facecolor="red", edgecolor="black", alpha=0.6, label="Wrong tile"),
            Patch(facecolor="none", edgecolor="steelblue", linewidth=2, label="Tile boundary"),
        ], loc="lower right", fontsize=10)

        # -----------------------------
        # TITLE + ACCURACY
        # -----------------------------
        n = len(meta_df)
        acc = 100 * correct_count / n if n > 0 else 0

        tile_txt = f" | tile: {focus_tile}" if focus_tile is not None else ""

        ax.set_title(
            title or f"Tile predictions — {split_name}{tile_txt} | accuracy: {correct_count}/{n} ({acc:.1f}%)",
            fontsize=13
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Map saved: {save_path}")

        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def _unscale_coordinates(self, coords_tensor):
        lat_min, lat_max = self.params["transfer"]["params"]["lat_bounds"]
        lon_min, lon_max = self.params["transfer"]["params"]["lon_bounds"]
        coords_unscaled = np.zeros_like(coords_tensor)
        coords_unscaled[..., 0, :, :] = coords_tensor[..., 0, :, :] * (lat_max - lat_min) + lat_min
        coords_unscaled[..., 1, :, :] = coords_tensor[..., 1, :, :] * (lon_max - lon_min) + lon_min
        return coords_unscaled

    @staticmethod
    def haversine_distance_numpy(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = (np.sin(delta_phi / 2.0) ** 2
             + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
        return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    def compute_pixel_errors_meters(self, split_name):
        loader = self.loaders[split_name]
        preds_scaled = self.results[split_name].numpy()
        all_labels = [loader.dataset[i][1] for i in range(len(loader.dataset))]
        labels_scaled = np.array(all_labels)
        preds_gps = self._unscale_coordinates(preds_scaled)
        labels_gps = self._unscale_coordinates(labels_scaled)
        print(labels_gps.shape)
        errors_meters = self.haversine_distance_numpy(
            labels_gps[:, 0], labels_gps[:, 1],
            preds_gps[:, 0],  preds_gps[:, 1],
        )
        return errors_meters, preds_gps, labels_gps

    def plot_geospatial_errors_distribution(self, errors_meters, save_path=None):
        flat_errors = errors_meters.flatten()
        median_err = np.median(flat_errors)
        mean_err = np.mean(flat_errors)
        p95 = np.percentile(flat_errors, 95)
        p99 = np.percentile(flat_errors, 99)
        plt.figure(figsize=(12, 6))
        plt.hist(flat_errors, bins=100, color="darkslateblue", edgecolor="black", alpha=0.7, log=True)
        plt.axvline(median_err, color="orange", linestyle="--", label=f"Median: {median_err:.1f} m")
        plt.axvline(mean_err,   color="red",    linestyle=":",  label=f"Mean: {mean_err:.1f} m")
        plt.title("Distribution of Localization Error per Pixel")
        plt.xlabel("Error (meters)")
        plt.ylabel("Pixels (log)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        print(f"Mean: {mean_err:.2f} m | Median: {median_err:.2f} m | P95: {p95:.2f} m | P99: {p99:.2f} m")

    def display_patch_error_map(self, split_name, k, errors_meters, preds_gps, labels_gps, save_path=None):
        loader = self.loaders[split_name]
        patch_id = loader.dataset.metadata_frame.iloc[k]["ID_PATCH"]
        inputs, *_ = loader.dataset[k]
        X_batch = inputs[0]
        img_rgb = X_batch.median(dim=0)[0][[2, 1, 0]].permute(1, 2, 0).numpy()
        img_rgb = np.clip((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8), 0, 1)
        patch_errors = errors_meters[k]
        h_mid, w_mid = patch_errors.shape[0] // 2, patch_errors.shape[1] // 2
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"RGB (ID: {patch_id})")
        axes[0].axis("off")
        im = axes[1].imshow(patch_errors, cmap="viridis")
        plt.colorbar(im, ax=axes[1], label="Error (m)")
        axes[1].set_title(f"Error Heatmap — Median: {np.median(patch_errors):.1f} m")
        axes[1].axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.show()
        plt.close(fig)

    def display_interactive_html_map(self, split_name, errors_meters, preds_gps, labels_gps,
                                      save_html_path, max_patches=100):
        loader = self.loaders[split_name]
        metadata_df = loader.dataset.metadata_frame
        center_lat = np.mean(labels_gps[:, 0])
        center_lon = np.mean(labels_gps[:, 1])
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
        indices = np.random.choice(len(metadata_df), min(max_patches, len(metadata_df)), replace=False)
        metadata_full = gpd.read_file(os.path.join(self.dataset_root, "metadata.geojson"))
        tile_layer = folium.FeatureGroup(name="Tile boundaries")
        tile_boundaries = (
            metadata_full.groupby("tile")["geometry"]
            .agg(lambda geoms: geoms.unary_union.envelope)
        )
        for tile_id, tile_geom in tile_boundaries.items():
            minx, miny, maxx, maxy = tile_geom.bounds
            folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]],
                             color="steelblue", weight=1.5, fill=False,
                             tooltip=f"Tile {tile_id}").add_to(tile_layer)
        tile_layer.add_to(m)
        folium.LayerControl().add_to(m)
        for idx in indices:
            patch_id = metadata_df.iloc[idx]["ID_PATCH"]
            h_mid = errors_meters.shape[1] // 2
            w_mid = errors_meters.shape[2] // 2
            folium.CircleMarker(
                location=[labels_gps[idx, 0, h_mid, w_mid], labels_gps[idx, 1, h_mid, w_mid]],
                radius=6, color="blue", fill=True, popup=f"TRUE — Patch {patch_id}",
            ).add_to(m)
            folium.CircleMarker(
                location=[preds_gps[idx, 0, h_mid, w_mid], preds_gps[idx, 1, h_mid, w_mid]],
                radius=6, color="red", fill=True,
                popup=f"PRED — Median err: {np.median(errors_meters[idx]):.1f} m",
            ).add_to(m)
            folium.PolyLine(
                locations=[[labels_gps[idx, 0, h_mid, w_mid], labels_gps[idx, 1, h_mid, w_mid]],
                           [preds_gps[idx, 0, h_mid, w_mid],  preds_gps[idx, 1, h_mid, w_mid]]],
                color="purple", weight=2.5, opacity=0.8,
                tooltip=f"Patch {patch_id} | Err: {np.median(errors_meters[idx]):.1f} m",
            ).add_to(m)
        os.makedirs(os.path.dirname(save_html_path), exist_ok=True)
        m.save(save_html_path)
        print(f"Interactive map saved: {save_html_path}")

    @staticmethod
    def plot_f0_distribution(tensor, title="f0 distribution",
                              bins=100, save_path=None, scale=None, abs=None, a=None, b=None):
        values  = tensor.detach().cpu().numpy().flatten()
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        p95_val = np.percentile(values, 95)
        p99_val = np.percentile(values, 99)
        pct = None
        if a is not None and b is not None:
            pct = np.mean((values >= a) & (values <= b)) * 100
        log = (scale == "log")
        plt.figure(figsize=(12, 6))
        plt.hist(values, bins=bins, color="teal", edgecolor="black", alpha=0.7, log=log)
        plt.axvline(mean_val,   color="red",    linestyle="dashed",  label=f"Mean: {mean_val:.4e}")
        plt.axvline(median_val, color="orange", linestyle="dotted",  label=f"Median: {median_val:.4e}")
        plt.title(title)
        plt.xlabel(abs)
        plt.ylabel("pixels (log)" if log else "pixels")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Figure saved: {save_path}")
        plt.show()
        print(f"Mean: {mean_val:.6f} | Std: {std_val:.6f} | Median: {median_val:.6f} "
              f"| P95: {p95_val:.6f} | P99: {p99_val:.6f}")
        if pct is not None:
            print(f"| % in [{a},{b}]: {pct:.2f}%")

    def display_heatmap(self, split_name, k, save_path=None):
        if self.results is None or split_name not in self.results:
            print("Error: results not loaded.")
            return
        loader = self.loaders[split_name]
        patch_id = loader.dataset.metadata_frame.iloc[k]["ID_PATCH"]
        f0_map = self.results[split_name][k, 1].numpy()
        print(f"k={k} | ID_PATCH: {patch_id} | f0 min: {f0_map.min():.2f} max: {f0_map.max():.2f} "
              f"pixels>17: {(f0_map > 17).sum()}")
        inputs, *_ = loader.dataset[k]
        X_batch = inputs[0]
        img_rgb = X_batch.median(dim=0)[0][[2, 1, 0]].permute(1, 2, 0).numpy()
        img_rgb = np.clip((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8), 0, 1)
        threshold = 17.0
        f0_max = max(f0_map.max(), threshold + 1.0)
        f0_scaled = np.clip((f0_map - threshold) / (f0_max - threshold), 0, 1)
        cmap = plt.get_cmap("autumn")
        overlay = cmap(f0_scaled).astype(np.float64)
        overlay[..., 3] = np.where(f0_map > threshold, 0.85, 0.0)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"RGB (ID: {patch_id})")
        axes[0].axis("off")
        axes[1].imshow(img_rgb)
        axes[1].imshow(overlay, interpolation="nearest")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=threshold, vmax=f0_max))
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1], label=f"f0 (> {threshold})")
        axes[1].set_title(f"Heatmap f0 > {threshold}")
        axes[1].axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"Saved: {save_path}")
        plt.show()
        plt.close(fig)

    def display_patches_on_map(self, split_name, patch_indices,
                               padding=0.5, title=None,
                               zoom_patch_idx=None, show_rgb=False):
        if self.results is None or split_name not in self.results:
            print("Error: results not loaded for this split.")
            return
        if not patch_indices:
            print("No patches to display.")
            return
        loader = self.loaders[split_name]
        metadata = gpd.read_file(os.path.join(self.dataset_root, "metadata.geojson"))
        france = self._load_france_boundaries()
        fig, ax = plt.subplots(figsize=(12, 12))
        france.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5)
        for k in patch_indices:
            patch_id = loader.dataset.metadata_frame.iloc[k]["ID_PATCH"]
            row = metadata[metadata["ID_PATCH"] == patch_id]
            if row.empty:
                continue
            minx, miny, maxx, maxy = row.iloc[0].geometry.bounds
            if show_rgb and zoom_patch_idx == k:
                inputs, *_ = loader.dataset[k]
                X_batch = inputs[0]
                img_rgb = X_batch.median(dim=0)[0][[2, 1, 0]].permute(1, 2, 0).numpy()
                img_rgb = np.clip((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8), 0, 1)
                ax.imshow(img_rgb, extent=[minx, maxx, miny, maxy], origin="upper", aspect="auto", zorder=5)
            else:
                ax.add_patch(plt.Rectangle(
                    (minx, miny), maxx - minx, maxy - miny,
                    linewidth=2, edgecolor="red", facecolor="red", alpha=0.6, zorder=5,
                ))
                ax.text((minx + maxx) / 2, maxy, str(patch_id),
                        fontsize=8, ha="center", va="bottom", color="red", zorder=6)
        if zoom_patch_idx is not None:
            patch_id = loader.dataset.metadata_frame.iloc[zoom_patch_idx]["ID_PATCH"]
            row = metadata[metadata["ID_PATCH"] == patch_id]
            minx, miny, maxx, maxy = row.iloc[0].geometry.bounds
            ax.set_xlim(minx - padding, maxx + padding)
            ax.set_ylim(miny - padding, maxy + padding)
        else:
            ax.set_xlim(-5.5, 9.5)
            ax.set_ylim(41.0, 51.5)
        ax.set_title(title or f"{split_name} — {len(patch_indices)} patch(s)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _load_france_boundaries():
        import requests, zipfile, io
        cache_dir = os.path.join("/home/alexian/data/hedgementation", "cache", "naturalearth")
        os.makedirs(cache_dir, exist_ok=True)
        shp_path = os.path.join(cache_dir, "ne_110m_admin_0_countries.shp")
        if not os.path.exists(shp_path):
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            r = requests.get(url, timeout=30)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(cache_dir)
        world = gpd.read_file(shp_path)
        return world[world["NAME"] == "France"]

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.results, os.path.join(save_dir, "inference_results.pt"))
        import pickle
        with open(os.path.join(save_dir, "inference_metadata.pkl"), "wb") as f:
            pickle.dump({k: v.dataset.metadata_frame for k, v in self.loaders.items()}, f)
        with open(os.path.join(save_dir, "inference_params.json"), "w") as f:
            json.dump(self.params, f)

        if hasattr(self, "majority_vote"):
            with open(os.path.join(save_dir, "tile_state.pkl"), "wb") as f:
                pickle.dump({
                    "majority_vote": self.majority_vote,
                    "entropy": self.entropy,
                    "labels": self.labels,
                }, f)

        print(f"Saved to {save_dir} — splits: {list(self.results.keys())}")

    @staticmethod
    def load(save_dir, dataset_root):
        import pickle
        results = torch.load(os.path.join(save_dir, "inference_results.pt"))
        with open(os.path.join(save_dir, "inference_metadata.pkl"), "rb") as f:
            metadata_frames = pickle.load(f)
        with open(os.path.join(save_dir, "inference_params.json"), "r") as f:
            params = json.load(f)

        transfer = deserialize_label_fn(params["transfer"])
        loaders = {
            split_name: setup_dataloader(
                metadata_frame=df,
                data_path=dataset_root,
                num_buckets=params["num_buckets"],
                image_count=params["image_count"],
                normalization=params["normalization"].upper(),
                y_transform=params["y_transform"].upper(),
                inclusion_intervals=params["inclusion_intervals"],
                batch_size=params["batch_size"],
                shuffle=False,
                transfer=transfer,
                load_X_cloud=True,
                cloud_threshold=params["cloud_threshold"],
                cloud_band=params["cloud_band"], 
            )
            for split_name, df in metadata_frames.items()
        }
        gen = DensityMapGenerator.__new__(DensityMapGenerator)
        gen.model_dir = save_dir
        gen.dataset_root = dataset_root
        gen.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen.params = params
        gen.model = None
        gen.results = results
        gen.loaders = loaders
        tile_state_path = os.path.join(save_dir, "tile_state.pkl")
        if os.path.exists(tile_state_path):
            import pickle
            with open(tile_state_path, "rb") as f:
                state = pickle.load(f)
            gen.majority_vote = state["majority_vote"]
            gen.entropy = state["entropy"]
            gen.labels = state["labels"]
        print(f"Loaded from {save_dir} — splits: {list(results.keys())}")
        return gen