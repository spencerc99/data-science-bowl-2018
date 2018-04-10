import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from models.config import Config
import utils
import models.rcnn_model as modellib
import visualize
from models.rcnn_model import log
from models import rcnn_util
from sklearn.model_selection import train_test_split
from constants import *
from imageio import imread
from skimage.transform import resize
from models.basic_model import BasicModel

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    rcnn_util.download_trained_weights(COCO_MODEL_PATH)

class RCNNConfig(Config):
    # Give the configuration a recognizable name
    NAME = "dsb"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 2

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32


class RCNNDataset(rcnn_util.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self):
        super().__init__()
        self.train_patients, self.val_patients = train_test_split(utils.get_patients(), test_size=.1, random_state=42)
        self.patients = []

    def load_images(self, type):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        assert type in ["val", "train"]
        # Add classes
        self.add_class("dsb", 1, "nuclei")

        if type == "val":
            self.patients = self.val_patients
        elif type == "train":
            self.patients = self.train_patients
            
        for _id in self.patients:
            self.add_image("dsb", image_id=_id, path=None,
                           width=IMG_WIDTH, height=IMG_HEIGHT)

    def load_image(self, image_id):
        image_id = self.patients[image_id]
        path = TRAIN_FOLDER + '/input/' + str(image_id)
        img = imread(path + '/images/' + str(image_id) + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True)
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        
        info = self.image_info[image_id]
        if info["source"] == "dsb":
            return info["dsb"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_id = self.patients[image_id]
        path = TRAIN_FOLDER + '/input/' + image_id
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

class RCNN(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.lr = config['lr']
        rcnn_config = RCNNConfig()
        self.model = modellib.MaskRCNN(mode="training", config=rcnn_config,
                                  model_dir=MODEL_DIR)
        self.init_with = config["init_with"] if "init_with" in config else "coco"  # imagenet, coco, or last

        if self.init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif self.init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            self.model.load_weights(COCO_MODEL_PATH, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif self.init_with == "last":
            # Load the last model you trained and continue training
            self.model.load_weights(self.model.find_last()[1], by_name=True)
    
    def train(self):
        dataset_train = RCNNDataset()
        dataset_train.load_images("train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = RCNNDataset()
        dataset_val.load_images("val")
        dataset_val.prepare()
        if self.init_with == "last":
            self.model.train(dataset_train, dataset_val,
                        learning_rate=self.lr/ 10,
                        epochs=2,
                        layers="all")
        else:
            self.model.train(dataset_train, dataset_val,
                        learning_rate=self.lr,
                        epochs=1,
                        layers='heads')
