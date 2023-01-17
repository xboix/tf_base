import json
import os
import numpy as np


def config_datasets(datasets_dir, create_json=True):

    id = 0
    dataset_list = []

    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "mnist"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1


    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "fashion_mnist"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1

    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "cifar"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1

    print(str(id) + " dataset config files created")
    return dataset_list


