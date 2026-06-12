import itertools
import json
import geopandas as gpd
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS",5))

TRAIN_FOLDS = json.loads(os.environ["TRAIN_FOLDS"] if os.environ["TRAIN_FOLDS"] else [0,1,2]) 
VALID_FOLDS = json.loads(os.environ["VALID_FOLDS"] if os.environ["VALID_FOLDS"] else [3])
TEST_FOLDS = json.loads(os.environ["TEST_FOLDS"] if os.environ["TEST_FOLDS"] else [4])

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
class CrossTileGroupAnalyzer:
    def __init__(self,
                 experiment_prefix: str,
                 model_directory: str = "models/",
                 experiment_suffix: str = "_far_group_{i}",
                 metrics=["precision", "recall", "f1", "iou"],
                 num_tilegroups=NUM_TILEGROUPS
                 ):
        self.experiment_prefix = experiment_prefix
        self.experiment_suffix = experiment_suffix
        self.model_directory = model_directory
        self.num_tilegroups=num_tilegroups
        self.experiment_directories = {
            i: os.path.join(
                self.model_directory, f"{self.experiment_prefix}{self.experiment_suffix.format(i=i)}"
                ) for i in range(self.num_tilegroups)
        }

        self.experiment_dataframes = {
            i: gpd.read_file(
                os.path.join(
                    self.experiment_directories[i],
                    "best_model_per_patch_performance.geojson"
                )
            ) for i in range(self.num_tilegroups)
        }

        self.experiment_metrics = {
            i: json.load(open(
                os.path.join(
                    self.experiment_directories[i],
                    "best_trained_model_metrics.json"
                )
            , "rb+")) for i in range(self.num_tilegroups)
        }

        self.metrics = metrics

        self.performance_as_near_dataframe = self.calculate_performance_as_near(
            self.experiment_dataframes
        )

        self.performance_as_far_dataframe = self.calculate_performance_as_far(
            self.experiment_dataframes
        )

        self.delta_near_to_far = self.calc_change_in_performance_from_near_to_far()

    def merge_helper(self,
                     df1: gpd.GeoDataFrame, 
                     df2: gpd.GeoDataFrame):
        concat = pd.concat([df1,df2])

        agg_dict = {col:"first" for col in df1.columns if col != "ID_PATCH"}
        for m in self.metrics:
            agg_dict[m] = lambda x: list(x)
        

        result = concat.groupby(by="ID_PATCH").aggregate(agg_dict).reset_index()
        return result
        

    def calculate_performance_as_near(self,
            dataframes: dict[int:gpd.GeoDataFrame]
    ):
        results = None

        for tile_group in dataframes:
            curr_dataframe = dataframes[tile_group]
            curr_dataframe = curr_dataframe[curr_dataframe["tile_group"] != tile_group]
            if results is None:
                results = curr_dataframe
            else:
                results = self.merge_helper(curr_dataframe, results)
        for m in self.metrics:
            results[m] = results[m].apply(
                lambda x: list(flatten(x))
            )


            results[f"mean_{m}"] = results[m].apply(
                lambda x: np.mean(np.array(x).flatten())
            )
        return results
                
    def calculate_performance_as_far(self,
            dataframes: dict[int:gpd.GeoDataFrame]
    ):
        results = None

        for tile_group in dataframes:
            curr_dataframe = dataframes[tile_group]
            curr_dataframe = curr_dataframe[curr_dataframe["tile_group"] != tile_group]
            if results is None:
                results = curr_dataframe
            else:
                results = pd.concat([results, curr_dataframe], ignore_index=True)

        return results
    
    def calc_change_in_performance_from_near_to_far(self):
        columns = {}

        for m in self.metrics:
            column = f"delta_{m}"

            delta = self.performance_as_near_dataframe[f"mean_{m}"] - self.performance_as_far_dataframe[m]
            columns[column] = delta
        
        columns["ID_PATCH"] = self.performance_as_far_dataframe["ID_PATCH"]
        columns["tile_group"] = self.performance_as_far_dataframe["tile_group"]
        delta_performance = pd.DataFrame(data=columns)
        return delta_performance