# Hedgementation Dataset Construction
This repo consists of all the code necessary to construct, export, download, and verify the Hedgementation dataset. This is performed in five broad steps.

1. Setup the environment and download the necessary dependencies and datasets.

2. Construct the base asset, which divides the entirety of France into "patches"(1280x1280m squares) with their associated hedgerows and metadata.

3. Create the hedgementation dataset by randomly sampling the base asset stratified across tiles.

4. Export the created dataset to a destination of your choice. 

5. Download the exported data to a local destination and verify it's integrity. 

## Step 1: Setup the Environment and Download Datasets

Firstly, download necessary Python packages and setup environment variables.

- Install python.

- Run ``python3 -m venv .venv``.

- Run ``source .venv/bin/activate`` if on Linux/Mac, or ``.venv/Scripts/Activate`` if on Windows.

- Run ``pip install -r requirements.txt``.

- Run ``pip install ../hedgementation_utils``. This will install a shared util repo for the Hedgementation project that is necessary for the rest of the project to run.

- Copy ``example.env`` into ``.env``, replacing the variables marked **TODO** with the appropriate values for your local setup.

Then download the external datasets. 

- Run ``bash scripts/download_bda.sh``, or copy the url from the first line into your browser and open the downloaded archive. Move the file ``haie.gpkg`` to an appropriate location. Set ``INITIAL_BDA_PATH`` in your ``.env`` to point to this file. 
- Run ``bash scripts/download_rpg.sh``, or copy the url from the first line into your browser and open the downloaded archive. Move the file ``PARCELLES_GRAPHIQUES.shp`` and it's sidecar files(``.dbf``, ``.prj``, and ``.shx``) from the resulting archive to an appropriate location. Set ``RPG_ASSET`` in your ``.env`` to the location of the ``.shp`` file.
- Run ``bash scripts/download_aez.sh``, or copy both urls and download and extract both archives. Move all of the files from ``LR/aez`` to a single location, and move the files from ``clr_files`` to the same location. Set ``AEZ_ROOT`` to point here.

## Step 2: Create the Base Asset
TL;DR - Run ``python3 -m src.construct_dataset.construct_base_asset --workers 8`` 

The base asset is created by dividing the entire breadth of the BDA hedgerows asset into constant-sized square "patches", and assigning hedgerows and metadata to each patch. Luckily, this can be done almost entirely automatically with very little interference using the file ``src/construct_dataset/construct_base_asset.py``. Simply running this script will construct the full base asset from nothing but the datasets downloaded in the previous step. It does so by dividing the scope of the enormous BDA dataset into a large number of "supertiles" that are all individually built out and re-merged into the full base asset. This is done using a concurrent worker-based approach, and we recommend settings ``--workers`` defaults to 1, we recommend using a higher number, such as 8 or 16, to significantly speed up the process.

The results for individual supertiles are saved into the provided ``output_dir`` to make the process resilient to crashes and interruptions. If anything is found in this directory when the script starts, it will pick up where it left off rather than re-doing any work. If this is undesirable, delete the contents of ``output_dir`` or point it to a different directory between runs.

### Reproducing the legacy tile and fold labels

By default the pipeline generates its tile and fold grids from scratch using deterministic, seeded procedures. The historical dataset iteration was built with grids that cannot be regenerated from first principles (its tile ordering depended on floating-point noise and its folds on an unseeded RNG), so those grids are preserved as data in ``configs/legacy_manifests/`` (``tile_grid.parquet``, ``cell_grid.parquet``, recovered once from the legacy asset by ``src/construct_dataset/build_legacy_manifests.py``). To build a base asset that matches the legacy iteration row-for-row, run into a fresh ``--output_dir`` with:

```
python3 -m src.construct_dataset.construct_base_asset --workers 8 --legacy_manifests_dir configs/legacy_manifests --save_crs EPSG:4326
```

Omit the flags to use the new assignment procedures instead. Notes on legacy mode:

- Patches without a fold are kept with ``fold=-1`` (the legacy convention) instead of being dropped.
- ``--save_crs EPSG:4326`` saves the patch geometry *and* the hedgerow WKT strings in degrees, like the legacy asset; without it both are in EPSG:3857 metres.
- The legacy fold cells sit on a 2-patch gap pitch while the new default is ``--patches_between_cells 3``; in legacy mode the cell grid is loaded from the manifest, so the pitch arguments are ignored.
- Do not round or alter ``--aoi_bounds``: the default is the full-precision origin of the legacy patch grid, and even a millimetre shift drops patches whose hedgerows overlap them by less than that.

The final dataset is always written sorted by ``(patch_i, patch_j)``, so its row order is independent of worker count and supertile layout (and matches the legacy asset's order).

## Step 3: Sample the Base Asset to create the Hedgementation Dataset

Several notebooks are available within ``notebooks/construct_dataset`` to create different versions of the dataset. The most up-to-date versions are found in ``create_dataset_1.3.ipynb`` and ``create_dataset_1.3.ipynb``. The two only differ in that 1.3 contains 4x as many datapoints as 1.2. 1.0 and 1.1 are also present for archiving purposes, but we don't recommend using either of these. Running either the 1.2 or 1.3 notebook will generate a GeoPandas GeoDataFrame file that can be used as a basis to generate and export the actual data in the dataset in the following step. You can edit either notebook to fit your needs as required.

## Step 4: Export the Hedgementation Dataset
TL;DR - Setup the GCloud CLI as explained [here](https://docs.cloud.google.com/sdk/docs/install-sdk). Then, familiarize yourself with the script's arguments, create a config file with your preferred arguments, and schedule a job to run ``python3 -m src.gee_export.full_dataset_export --config [YOUR_CONFIG_HERE]`` at regular intervals until the dataset is fully exported. An example config is available in ``configs/full_dataset_export`` which should work with minimal changes. The contents of the exported data (date range, satellite collections and bands, and the downsampling procedure used to generate y) are controlled by a second set of JSON configs in ``configs/gee_export/``, described under "Export configuration files" below; the defaults reproduce the published dataset and don't need to be touched for a standard export.

Once you have created the dataset, it has to be exported. The full exporting pipeline is centralized into a single script found in ``src/gee_export/full_dataset_export.py``. Since this pipeline downloads data using Google Earth Engine, you'll have to set up the [GCloud CLI on your local environment](https://docs.cloud.google.com/sdk/docs/install-sdk), create a Google Cloud project and enable Earth Engine access on your own Google account, then copy the ID of that project to GCLOUD_PROJECT_ID in your .env file. You will also likely have to set up a service account using [service account impersonation](https://docs.cloud.google.com/docs/authentication/use-service-account-impersonation), generate a key, and copy it to your local environment(the script looks for it in ``secrets/service_account_key.json``). Google Cloud authentication is very finnicky for these sorts of projects, so you may run into some difficulties here. 

Once Google Cloud is set up, you can now run the pipeline, passing the path to the saved GeoDataFrame from step 3 as ``--source``.  The file will open the GeoDataFrame passed as ``source``, and use the geometry of the row, and the hedgerow multilinestring, to generate the necessary data on GEE. By default it will export each of the following for every datapoint in ``source``. If you want to skip any of these, pass them as one of the arguments to ``--skip``. 
- **X / satellite**: An image time-series of 10-band Sentinel 2 satellite images, created at a spatial resolution of 10m/pixel, resulting in 128x128 pixel images. These are the primary features used to train and run inference with our model. By default, images beyond a certain cloud threshold (20% cloudy pixels, set in ``configs/gee_export/sentinel2.json``) are filtered out, but this can be avoided by passing ``--no_cloud_filter`` when running the script. The collection, bands, date range, resolution and image size are all set by the configs described below.
- **y / hedgerows**: A single 128x128 pixel image representing the groud-truth hedgerow labels, where each pixel contains a continuous value in [0,1], where a value closer to 1 means more hedgerow presence. These labels are created by buffering the lines of the hedgerow multilinestring geometry to a width of 7 meters and rasterizing it at a spatial resolution of 0.5 meters/pixel. It is then converted to a resolution of 10 meters/pixel by averaging the values of every 20x20 pixel square. All of these parameters can be changed via ``configs/gee_export/downsampling.json``. We do not directly train on the continuous values of y, instead using them as a means to assign class labels to each pixel(i.e. non-0 pixels are "hedgerow", 0 pixels are background), but having access to the raw continuous values is still useful.
- **X_cloud / cloud**: An image time-series constructed with the same dimensions and across the same time interval as X, but sampled from the Cloud Score+ metrics instead. Our current approach is to export this alongside X, and use it at load time to filter out cloudy timesteps. 
- **y_id / identity**: A 128x128 raster that distinguishes hedgerow from background, while also identifying which individual linestring each pixel belongs to within the larger hedgerow multilinestring. Background pixels are given a value of ``-1``, while pixels that belong to the ``n``th linestring has a value of ``n``. 
- **rpg**: A 128x128 binary mask distinguishing background from non-background pixels according to the RPG data. These are downloaded directly to your local drive, so they'll require you to set ``DATASET_ROOT`` and ``RPG_SUBDIR``(the root folder for your version of the dataset, and the subdirectory within where you want to store RPG masks respectively.)
- **aef**: An embedding of the provided patch over the full date range generated by Google's Alpha Earth Foundation models, returned in 64x128x128 format. 


All of these values are exported by default, but one or more can be skipped by passing the respective value to the ``--skip_tasks`` flag in the script.

### Export configuration files

While the main ``--config`` file controls *where* data is exported (buckets, folders) and how the run behaves, the parameters governing *what* is exported live in a second set of JSON config files under ``configs/gee_export/``. These are loaded automatically, and their defaults match the values used to export the published dataset, so for a standard export you don't need to touch them. Each one can be re-pointed to a different file via a CLI flag, or by setting the same key inside your main ``--config`` JSON:

- ``--shared_export_config`` (``shared_export.json``): settings shared by all satellite time-series exports — the date range (``start_date``/``end_date``, ``YYYY-MM-DD``), ``crs``, ``scale`` (m/pixel), and ``nb_pixel`` (output image size). Defaults: 2021-09-17 to 2022-10-27, EPSG:3857, 10m/pixel, 128x128.
- ``--satellite_config`` (``sentinel2.json``): the Sentinel-2 ``collection``, ``selected_bands``, and ``cloud_threshold`` (max ``CLOUDY_PIXEL_PERCENTAGE`` when cloud filtering, default 20). Dates always come from the shared config.
- ``--cloud_score_config`` (``cloud_score.json``): the Cloud Score+ ``collection`` and ``selected_bands``. May optionally specify its own ``start_date``/``end_date``; when omitted it inherits the shared date range (recommended, since X_cloud is meant to align with X).
- ``--aef_config`` (``aef.json``): the AEF embedding ``collection`` and ``windows``, a list of ``[start, end]`` date pairs whose annual embeddings are averaged. Defaults to the 2020-21 and 2021-22 agronomic years. When ``windows`` is omitted, a single window spanning the shared date range is used.
- ``--downsampling_config`` (``downsampling.json``): the procedure used to generate y/y_id — hedgerow buffer ``width`` (meters, default 7), ``initial_scale`` (m/pixel of the high-res rasterization, default 0.5), ``reduce_resolution``, and optionally ``final_scale`` (inherits the shared ``scale`` when omitted).

For example, to export over a different date range, copy the shared config, edit the dates, and point the script at your copy. With a file like ``configs/gee_export/my_dates.json``:

```json
{
    "start_date": "2022-09-17",
    "end_date": "2023-10-27",
    "crs": "EPSG:3857",
    "scale": 10,
    "nb_pixel": 128
}
```

``python3 -m src.gee_export.full_dataset_export --config [YOUR_CONFIG_HERE] --shared_export_config configs/gee_export/my_dates.json``

or equivalently, add ``"shared_export_config": "configs/gee_export/my_dates.json"`` to your main config file. Unknown keys in any of these files are rejected with an error, so typos won't silently fall back to defaults. The standalone AEF exporter ``src/gee_export/aef1_3_export.py`` accepts ``--aef_config`` and ``--shared_config`` flags pointing to the same files.


We source our data from Google Earth Engine which does not allow for direct exporting to a local drive. This means that data must first be exported to a cloud destination(Either a google cloud storage bucket or folders in google drive) and then downloaded to your local storage. WARNING: Be aware that Google Cloud Storage charges egress fees for every gigabyte downloaded from their storage. Depending on the amount of data, this can end up racking up pretty significant charges. We recommend exporting to Google Drive by default and only using Google Cloud Storage if necessary. If you download to Google Cloud, the script requires you to specify individual cloud bucket paths for each of the different types of data you want to export, or individual google drive folder IDs.

The task queue for GEE is limited to 3000 tasks at a time, and each datapoint in the dataset requires a minimum of 2 export tasks(X and y), and may need up to 5. That means multiple runs of the export script, split out over the course of days to give the queue time to process, are likely to be necessary.

For this reason, we recommend the following approach. Firstly, figure out the arguments you want to run the script with, and write them into a JSON config in ``configs/full_dataset_export``(An example config used to export the 1.3 dataset is provided). Then, schedule a job with your OS' job scheduler(i.e. cron on Linux, launchd on Mac, or Windows Task Schedular on Windows) to run the following at 12- to 24-hour intervals:

``python3 -m src.gee_export.full_dataset_export --config configs/full_dataset_export/[YOUR_CONFIG_HERE].json``

The script will automatically skip exports for data that has already been exported or is queued in the task queue, and will automatically stop executing once the queue is filled. That means all you need to do is re-run the script at regular intervals to refill the queue, and your whole dataset will be cleanly exported to your chosen destination.

## Step 5: Download and verify the Hedgementation Dataset

Once you have exported the dataset properly, you'll need to download it to a local destination to train on it. You'll have to create an [OAuth Client ID in your project](https://support.google.com/cloud/answer/15549257), download the ``client_secrets.json``, and place it in the root of the repository. Once you've done this, the code to download the data can be found in ``src/gee_export/download_exported_data.py``. The easiest way is to simply specify ``--source_root``, which is either the root path in a GCS storage bucket, or folder ID of the root Google Drive folder, where your export of the dataset is stored. Then specify ``--destination_root`` to point to the local folder that you want to download the data to. The script will automatically parse the top-level sub-directories within ``source_root``, create equivalent sub-directories in ``destination_root``, and download the files to that sub-directory with the appropriate suffices. For example, if ``source_root`` contains a single sub-directory called ``X``, the script will create an equivalent ``X`` folder inside of ``destination_root``, and download the files as ``X_0.tif``, ``X_1.tif``, etc. 

Once you've downloaded the full dataset, copy the geojson file that you used for the export in Step 4 to ``destination_root`` and rename it to ``metadata_geojson``. Then, you can verify the integrity of your files by running ``python3 -m src.dataset_verifier {YOUR_DATASET_ROOT_HERE}``. This will read all the patch IDs from your metadata.geojson, check the existence of all of the required files, and then open them to verify their integrity(ensuring raster shape is as expected, etc.). Once this passes without any errors you're done! 