import tensorflow as tf

import os
import data.input_data as input_data
from timeit import default_timer as timer
from networks.network import get_network
from datetime import datetime
import sys
import pickle
import zipfile
import hashlib


def train(config):

    if config["skip"]:
        print("SKIP")
        return

    # Setting up training parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    max_num_training_steps = config['max_num_training_steps']
    num_output_steps = config['num_output_steps']
    backbone_name = config['backbone']

    if os.path.isfile(config["model_dir"] + '/results/training.done') and not config["restart"]:
        print("Already trained")
        return


    # Setting up the data and the model
    data = input_data.load_data_set(results_dir=config['results_dir'], data_set=config['data_set'],
                                    standarized=config["standarize"], multiplier=config["standarize_multiplier"],
                                    reshape=config['reshape'], seed=seed)
    num_features = data.train.images.shape[1]

    model = get_network(backbone_name, config, num_features)

    model_dir = config['model_dir']
    start_iteration = 0
    if not os.path.exists(model_dir + '/checkpoints/'):
        os.makedirs(model_dir + '/checkpoints/')
        os.makedirs(model_dir + '/results/')
    elif config["restart"]:
        print("Restart training")
        import shutil
        shutil.rmtree(model_dir + '/checkpoints', ignore_errors=True)
        shutil.rmtree(model_dir + '/results', ignore_errors=True)
        shutil.rmtree(model_dir + '/Validation', ignore_errors=True)
        shutil.rmtree(model_dir + '/Train', ignore_errors=True)
        os.makedirs(model_dir + '/checkpoints/')
        os.makedirs(model_dir + '/results/')
    elif not len(os.listdir(model_dir + '/checkpoints/')) == 0:
        start_iteration = int(tf.train.latest_checkpoint(model_dir + '/checkpoints/').split('/')[-1])
        print("Reload existing checkpoint " + str(start_iteration))
        model.load_all(tf.train.latest_checkpoint(model_dir + '/checkpoints/'))

    # Initialize the summary writer, global variables, and our time counter.
    summary_train = tf.summary.create_file_writer(model_dir + "/Train")
    summary_val = tf.summary.create_file_writer(model_dir + "/Validation")
    sys.stdout.flush()

    # Main training loop
    training_time = 0.0
    training_time_history = []
    for ii in range(start_iteration, max_num_training_steps):
        x_batch, y_batch = data.train.next_batch(batch_size)

        if ii % num_output_steps == 0:
            print('\n Step {} {}:    ({})\n'.format(model.optimizer.iterations.numpy(), ii, datetime.now()))
            sys.stdout.flush()

            # Setting up data for testing and validation
            x_test, y_test = data.validation.next_batch(batch_size)

            model.set_mode('test')

            model.evaluate(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64), step=ii, summary=summary_train)
            model.evaluate(tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64), step=ii,  summary=summary_val)

            if training_time != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time_history.append(num_output_steps * batch_size / training_time)
                training_time = 0.0

                model.save_all(model_dir + '/checkpoints/' + str(ii))

            sys.stdout.flush()

        # Training step
        start = timer()

        model.set_mode('train')
        model.train_step(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64))
        end = timer()
        training_time += end - start

    # Flag the training completed and store the training time profile
    open(model_dir + '/results/training.done', 'w').close()
    with open(model_dir + '/results/training_time.pkl', 'wb') as f:
        pickle.dump(training_time_history, f)

