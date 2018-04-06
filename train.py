'''
Source: https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
'''
import os
import json
import tensorflow as tf
from utils import *

# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
from models import make_model

flags = tf.app.flags
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')

flags.DEFINE_string('model_name', 'UNet', 'Unique name of the model')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('initial_mean', 0., 'Initial mean for NN')
flags.DEFINE_float('initial_stddev', 1e-2, 'Initial standard deviation for NN')
flags.DEFINE_float('lr', 1e-3, 'The learning rate of SGD')
flags.DEFINE_float('nb_units', 20, 'Number of hidden units in Deep learning agents')

flags.DEFINE_integer('max_iter', 2000, 'Number of training step')
flags.DEFINE_integer('random_seed', 42, 'Value of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()
    # fixed_params must be a string to be passed in the shell, let's use JSON
    # config["fixed_params"] = json.loads(config["fixed_params"])

    model = make_model(config)
    
    results = model.train()

if __name__ == '__main__':
    tf.app.run()