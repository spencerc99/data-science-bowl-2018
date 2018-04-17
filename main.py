'''
Source: https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
'''
import os
import json
import tensorflow as tf
from utils import *
import sys
# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
from models import make_model
from create_submission import create_submission_file, create_submission_file_rcnn

DEFAULT_LOGS_DIR = "/logs/"

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--model_name', required=True,
                        metavar="name of the model to train",
                        help='Should be one of \"RCNN\" or \"UNet\"')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--lr', required=False,
                         default=1e-3,
                         metavar="learning rate",
                         help="Learning rate")
    parser.add_argument('--ensemble', required=False,
                        default=False,
                        metavar="ensemble option",
                        help="ensemble option")
    args = parser.parse_args()
    weights = None if args.weights in ['coco', 'imagenet'] else args.weights
    config = {
        'model_name': args.model_name,
        'lr': 1e-3,
        'max_iter': 2000,
        'random_seed': 42,
        'init_with': 'last' if args.weights not in ['coco', 'imagenet'] else args.weights,
        'weights_path': weights,
        'type': args.command,
        'ensemble': bool(args.ensemble)
    }

    model = make_model(config)
    if args.command == "train":
        model.train()
        # model.predict()
    elif args.command == "predict":
        if not args.weights:
            raise Exception("Weights file needs to be passed for prediction")
        model.predict()
