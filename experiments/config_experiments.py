import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./experiments/base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    for datasets in range(3):
        config = base_config.copy()
        config["model_name"] = str(id)
        config["data_set"] = datasets
        config["backbone"] = "CNN"
        config["training_batch_size"] = 32
        config["standarize"] = False
        config["standarize_multiplier"] = 1.0
        if create_json:
            with open(results_dir + 'configs/' + str(id) + '.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1

    print(str(id) + " config files created")
    return experiment_list

