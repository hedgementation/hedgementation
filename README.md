# Hedgementation Benchmark

This repo contains all of the code necessary to build and use the [Hedgementation benchmark](http://arxiv.org/abs/2606.23615), train and evaluate baseline models, and train your own models on it.

The repository contains 3 subfolders.

# Dataset Construction

Contains all of the code needed to fully create the dataset from scratch, as well as instructions on how to do so in its README.md.

# Model Training

Contains the full scaffolding for training models on the Hedgementation dataset, as well as a quickstart guide in the README.md.

# Hedgementation Utils

A repo of util functionality shared across both other projects.

## How to Use

Start off in ``dataset_construction``. Follow the instructions to construct, export, and download the dataset. Afterwards, move to ``model_training``, which directs you on how to train a model on your local data.

# Reference

```bibtex
@inproceedings{senyard2026hedgementation,
  title     = {Hedgementation = Hedgerow Segmentation: A Remote Sensing Benchmark},
  author    = {Senyard, Nathan and Hamdani, Salem and Zhang, Astrid and Shelhamer, Evan and L{\'e}cuyer, Mathias and Gantois, Jos{\'e}phine},
  booktitle = {4th ICLR Workshop on Machine Learning for Remote Sensing (Main Track)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=mOMTBBgq5n}
}
