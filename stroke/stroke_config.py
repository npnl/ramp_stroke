import os
from os.path import join
import numpy as np
from bids.exceptions import BIDSValidationError
import bids

bids.config.set_option("extension_initial_dot", True)  # bids warning suppression

from stroke.bids_loader import BIDSLoader

estimator_filename = "estimator.py"
data_path = "data/"
# Used for checking that the data was downloaded correctly:

# Data
data_types = {"data": np.float32, "target": bool}

training = {
    "batch_size": 5,
    "root_dir": join(data_path, "train"),
    "data_entities": [{"subject": "", "session": "", "suffix": "T1w"}],
    "target_entities": [{"label": "L", "desc": "T1lesion", "suffix": "mask"}],
    "data_derivatives_names": ["ATLAS"],
    "target_derivatives_names": ["ATLAS"],
    "label_names": ["not lesion", "lesion"],
}

cross_validation = {"n_splits": 5, "train_size": 0.6, "random_state": 9001}

testing = {
    "root_dir": join(data_path, "test"),
    "batch_size": training["batch_size"],
    "data_entities": [{"subject": "", "session": "", "suffix": "T1w"}],
    "target_entities": [{"label": "L", "desc": "T1lesion", "suffix": "mask"}],
    "data_derivatives_names": ["ATLAS"],
    "target_derivatives_names": ["ATLAS"],
    "label_names": ["not lesion", "lesion"],
}

is_quick_test = "RAMP_TEST_MODE" in os.environ.keys()
num_subjects_quick_test = 5

try:
    if not os.path.exists(training["root_dir"]):
        train_path = join(os.path.dirname(__file__), data_path, "train")
        if os.path.exists(train_path):
            print(f"Changing training data path to {train_path}")
            training["root_dir"] = train_path
    if os.path.exists(training["root_dir"]):
        bids_loader_train = BIDSLoader(
            root_dir=training["root_dir"],
            data_entities=training["data_entities"],
            target_entities=training["target_entities"],
            data_derivatives_names=training["data_derivatives_names"],
            target_derivatives_names=training["target_derivatives_names"],
            label_names=training["label_names"],
            batch_size=training["batch_size"],
        )
        if is_quick_test:
            bids_loader_train.data_list = bids_loader_train.data_list[
                :num_subjects_quick_test
            ]
            bids_loader_train.target_list = bids_loader_train.target_list[
                :num_subjects_quick_test
            ]
except BIDSValidationError:
    print(
        "Warning: BIDS default path is not valid. Consider modifying config.py to match your data structure."
    )
    # default training path not valid; ignore
    pass

try:
    if not os.path.exists(testing["root_dir"]):
        test_path = join(os.path.dirname(__file__), data_path, "test")
        if os.path.exists(test_path):
            print(f"Changing training data path to {test_path}")
            testing["root_dir"] = test_path
    if os.path.exists(testing["root_dir"]):
        bids_loader_test = BIDSLoader(
            root_dir=testing["root_dir"],
            data_entities=testing["data_entities"],
            target_entities=testing["target_entities"],
            data_derivatives_names=testing["data_derivatives_names"],
            target_derivatives_names=testing["target_derivatives_names"],
            label_names=testing["label_names"],
            batch_size=testing["batch_size"],
        )
        if is_quick_test:
            bids_loader_test.data_list = bids_loader_test.data_list[
                :num_subjects_quick_test
            ]
            bids_loader_test.target_list = bids_loader_test.target_list[
                :num_subjects_quick_test
            ]
except BIDSValidationError:
    print(
        "Warning: BIDS default path is not valid. Consider modifying config.py to match your data structure."
    )
    # default testing path not valid; ignore
    pass
