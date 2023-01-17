"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import pickle
import tensorflow as tf
import sys
import os

from data import input_data
from networks.network import get_network


def test(config):

    # Setting up testing parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    backbone_name = config['backbone']

    if not os.path.isfile(config["model_dir"] + '/results/training.done'):
        print("Not trained")
        return

    if os.path.isfile(config["model_dir"] + '/results/testing.done') and not config["restart"]:
        print("Already tested")
        return

    # Setting up the data and the model
    data = input_data.load_data_set(results_dir=config['results_dir'], data_set=config['data_set'],
                                    standarized=config["standarize"], multiplier=config["standarize_multiplier"],
                                    reshape=config['reshape'], seed=seed)

    num_features = data.test.num_features
    model = get_network(backbone_name, config, num_features)
    model.load_all(tf.train.latest_checkpoint(config['model_dir'] + '/checkpoints/'), load_optimizer=False)

    model.set_mode('test')


    for dataset in ["val", "test"]:
        acc = []
        keep_testing = True
        while keep_testing:
            if dataset == "val":
                x_batch, y_batch = data.validation.next_batch(batch_size)
                keep_testing = not data.validation.last_iteration
            else:
                x_batch, y_batch = data.test.next_batch(batch_size)
                keep_testing = not data.test.last_iteration

            accuracy = model.evaluate(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64)).numpy()

            acc.append(accuracy)

        acc = np.mean(acc)
        print(acc)
        with open(config['model_dir'] + '/results/acc_' + dataset + '.pkl', 'wb') as f:
            pickle.dump(acc, f)


    open(config['model_dir'] + '/results/testing.done', 'w').close()




