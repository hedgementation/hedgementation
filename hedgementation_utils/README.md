## Hedgementation Utils
A set of utils for shared use across the hedgementation project.

## How to Install

Intended to be installed as a python package. Install using one of the following commands:

``pip install git+ssh://github.com/hedgementation/hedgementation_utils``

``pip install git+https://github.com/hedgementation/hedgementation_utils``

Make sure to at least set the DATASET_ROOT environment variable to point to your local version of the Hedgementation dataset.

Other local variables that can be optionally set:

| Field Name      | Description | Default |
| ----------- | ----------- | ----------- |
| TRAIN_FOLDS | A string representing the list of folds to use for training. Read into a string with json.loads(). | "[0,1,2]"
| VALID_FOLDS | A string representing the list of folds to use for validation. Read into a string with json.loads(). | "[3]"
| TEST_FOLDS | A string representing the list of folds to use for validation. Read into a string with json.loads(). | "[4]"
| NUM_TILEGROUPS | The number of tilegroups that the code assumes the patches are split into. | 5
| HEDGEMENTATION_RANDOM_SEED | The random seed used to seed random operations, such as dataset downsampling. | 15


You can also copy the contents of the provided example.env into your own .env file.