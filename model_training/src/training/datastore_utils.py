import pandas as pd
import numpy as np
import os
import json
import itertools

import torch

from dotenv import load_dotenv
load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]

TRAIN_DATA_PATH = os.path.join(DATASET_ROOT, "train_metadata.csv")
QUANTILE_DATA_PATTERN = os.path.join(DATASET_ROOT, "y/y_{}.npy") 
NORM_VALS_PATH = os.path.join(DATASET_ROOT, "normalization_vals.json")
def load_npy(path):
    with open(path, "rb+") as infile:
        return np.load(infile)

def load_quantile_data(path, pattern):
    test_data = pd.read_csv(path)
    ids = test_data["id"]
    def load_npy(id):
        with open(pattern.format(id), "rb+") as infile:
            return np.load(infile)

    loaded = np.array([load_npy(id) for id in ids])
    return loaded

def load_norm_vals(file_path, num_channels, norm_dims=(0,1,3,4)):
    if os.path.exists(file_path):
        with open(file_path,"rb+") as infile:
            norm_vals = json.load(infile)
            for k in norm_vals:
                norm_vals[k] = torch.tensor(norm_vals[k])
            return norm_vals
    else:
        train_metadata = DataStore.get_data(
            "train_metadata",
            load_func=lambda : TRAIN_DATA_PATH,
        )
        data_path = DataStore.get_data(
            "data_path",
            load_func=lambda : DATASET_ROOT,
        )
        from src.training.hedgementation_dataset import HedgementationDataset
        dataset = HedgementationDataset(train_metadata, data_path, constant_length=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

        mins = []
        maxs = []
        running_sum = torch.zeros(num_channels)
        n_samples = 0

        for (X,_),_ in dataloader:
            X = X.float()
            mins.append(X.amin(dim=norm_dims))
            maxs.append(X.amax(dim=norm_dims))
            running_sum += X.sum(dim=norm_dims)
            n_samples += np.prod([X.size(i) for i in norm_dims])

        mins = torch.stack(mins).min(0).values
        maxs = torch.stack(maxs).max(0).values
        means = running_sum / n_samples


        reshape_dims = [1 if i in norm_dims else -1 for i in range(X.dim())]
        means_reshaped = means.view(reshape_dims)
        sum_of_squares = torch.zeros(num_channels)
        for (X,_),_ in dataloader:
            X = X.float()
            sum_of_squares += ((X - means_reshaped) ** 2).sum(dim=norm_dims)

        stds = torch.sqrt(sum_of_squares / (n_samples - 1))

        with open(file_path, "w+") as outfile:
            json.dump({
                "min":mins.tolist(),
                "max":maxs.tolist(),
                "mean":means.tolist(),
                "std":stds.tolist()
            },outfile,indent=4)



        return {
            "mean":means,
            "min":mins,
            "max":maxs,
            "std":stds,
        }




class DataStore:
    _data = {}
    @classmethod
    def get_data(cls,name, load_func, load_func_args=None):
        if name not in cls._data:
            if load_func_args:
                loaded = load_func(**load_func_args)
            else:
                loaded = load_func()
            cls._data[name] = loaded
        return cls._data[name]

    @classmethod
    def set_data(cls,name,val):
        cls._data[name] = val