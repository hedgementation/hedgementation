# Hedgementation - Model Training

This is the central repo for training models on the Hedgementation dataset. Intended to be used after you have either constructed or downloaded the Hedgementation dataset to your local machine.

Owing to the size of the data, training consumes a substantial amount of GPU resources. For this reason, it's recommended to either train on a machine with access to a large amount of VRAM(such as a remote compute cluster) or to override the image_count to a number that fits within limited resoureces(such as 10).

## Quickstart

This tutorial assumes that you have already downloaded and set up the Hedgementationd dataset. If you have not, view the instructions on how to do so [here](TODO).

1. Copy `example.env` to `.env`, filling in `DATASET_ROOT`, `MODEL_DIR` and `CACHE_DIR` to point to the appropriate local directories.

2. Run `pip install -r requirements.txt`.

3. Run `pip install ../hedgementation_utils`. This will install the project's dedicated utils repo which is necessary for the code to run.

4. Run `python3 run.py --keywords test` to ensure the code runs properly. 
 
## Replicating Different Experiments

All of the experiments described in our paper correspond to pre-registered configs that can be run using the appropriate keywords in run.py. To run them, simply change the argument passed to `--keywords` in `run.py`.

- **Default model training**: `default`
- **Near vs. Far**: `far_group_0,far_group_1,far_group_2,far_group_3,far_group_4,`
- **Temperate vs. Subtropic Generalization**: `train_temperate,train_subtropic,downsample_to_subtropic`
- **Agricultural vs. Non-Agricultural**: `agriculture_mask_unbuffered,agriculture_mask_buffered_10m,downsample_to_unbuffered`

## Training

The full training loop is defined in ``src/training/trainer.py``, while all the information needed to replicate an experiment is captured in the ``TrainerConfig`` class found in ``src/training/trainer_config.py``. 

## Backbones

We trained models on a variety of different backbones, sourcing implementations from [PASTIS](https://openaccess.thecvf.com/content/ICCV2021/papers/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.pdf), a paper that trains on a similar task of image segmentation from satellite time-series. All of these implemenations can be found in the ``backbones/`` folder. We primarily used the ``UTAE`` model found in ``backbones/utae.py``, and would recommend this as the model to be used for experiments.


## Credits

 - Implementations for model backbones are credited to the authors of [PASTIS](https://github.com/VSainteuf/utae-paps) and [Fields of the world](TODO)
 - Our data is sourced from several different places. 
    - The satellite image timeseries used as our features are downloaded from the [Copernicus S2 Satellite via Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED). 
    - Our hedgerow labels are sourced from the 2024 version of the [BD Haie dataset](https://www.data.gouv.fr/datasets/bd-haie) released by the French government. 
    - AEF embeddings are sourced from the eponymous [Alpha Earth Foundations project](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/). 
    - Agricultural masks are created from the [IGNF RPG dataset](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_RPG) of agricultural parcels across France.
    - Climate metadata are sourced from the [Global Agro-Ecological Zones](https://gaez.fao.org/pages/data-access-download) dataset released by the Food and Agriculture Organization of the United Nations.
