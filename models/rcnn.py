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
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.model_selection import KFold
import datetime

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
    BACKBONE = "resnet50"

    STEPS_PER_EPOCH = (657 * .9) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 657 * .1 // IMAGES_PER_GPU)

    DETECTION_MIN_CONFIDENCE = 0

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    RPN_NMS_THRESHOLD = .95


class RCNNDataset(rcnn_util.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self, patients=None):
        super().__init__()
        if not patients:
            self.train_patients, self.val_patients = train_test_split(utils.get_patients(), test_size=.1, random_state=42)
        else:
            self.train_patients, self.val_patients = patients
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
        width, height, _ = imread(path + '/images/' + image_id + '.png').shape
        mask = []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file).astype(bool)
            # mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
            #                               preserve_range=True), axis=-1)
            mask.append(mask_)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(
            shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)
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
        self.weights_path = config['weights_path']
        self.ensemble = config['ensemble']
        self.init_with = config['init_with']
        self.start_ensemble = config['start_ensemble']
        if config['type'] == 'train':
            rcnn_config = RCNNConfig()
            self.model = self.make_model("training", rcnn_config, MODEL_DIR)
        else:
            class InferenceConfig(RCNNConfig):
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
            inference_config = InferenceConfig()
            self.model = self.make_model('inference', inference_config, MODEL_DIR)
    
    def make_model(self, mode, config, model_dir, weights_path=None):
        if not weights_path:
            weights_path = self.weights_path
        model = modellib.MaskRCNN(mode=mode, config=config,
                                  model_dir=model_dir)
        if weights_path:
            model.load_weights(weights_path, by_name=True)
        else:
            if self.init_with == "imagenet":
                model.load_weights(
                    model.get_imagenet_weights(), by_name=True)
            elif self.init_with == "coco":
                # Load weights trained on MS COCO, but skip layers that
                # are different due to the different number of classes
                # See README for instructions to download the COCO weights
                model.load_weights(COCO_MODEL_PATH, by_name=True,
                                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                 "mrcnn_bbox", "mrcnn_mask"])
            elif self.init_with == "last":
                # Load the last model you trained and continue training
                print("Training RCNN with last")
                model.load_weights(
                    model.find_last()[1], by_name=True)

        return model
        
    def get_last_ensemble(self):
        return [file for file in os.listdir(MODEL_DIR) if 'rcnn_ensemble' in file][-1]

    def get_last_weight(self, ensemble_dir):
        last_kfold = ensemble_dir + "/" + os.listdir(ensemble_dir)[-1]
        last_epoch = last_kfold + "/" + os.listdir(last_kfold)[-1]
        return last_epoch

    def train(self):
        if not self.ensemble:
            dataset_train = RCNNDataset()
            dataset_train.load_images("train")
            dataset_train.prepare()

            # Validation dataset
            dataset_val = RCNNDataset()
            dataset_val.load_images("val")
            dataset_val.prepare()
            self.train_model(self.model, dataset_train, dataset_val)
        else:
            model_dir = MODEL_DIR + "/rcnn_ensemble{:%Y%m%dT%H%M}".format(datetime.datetime.now())
            if self.start_ensemble > 0:
                model_dir = MODEL_DIR + "/" + self.get_last_ensemble()
                flag = True
            rcnn_config = RCNNConfig()
            k = 5
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            patients = np.array(utils.get_patients(), dtype=object)
            i = 1
            for train_idx, test_idx in kf.split(patients):
                print("Training model {} of ensemble".format(i))
                if i - 1 < self.start_ensemble:
                    i+=1
                    continue
                if flag:
                    model = self.make_model('training', rcnn_config, model_dir, self.get_last_weight(model_dir))
                    flag = False
                else:
                    model = self.make_model('training', rcnn_config, model_dir)
                dataset_train = RCNNDataset(patients=(patients[train_idx], patients[test_idx]))
                dataset_train.load_images("train")
                dataset_train.prepare()
                dataset_val = RCNNDataset(patients=(patients[train_idx], patients[test_idx]))
                dataset_val.load_images("val")
                dataset_val.prepare()
                
                self.train_model(model, dataset_train, dataset_val, epochs_head=10, epochs_tail=20)
                i+=1
    
    def train_model(self, model, dataset_train, dataset_val, epochs_head=20, epochs_tail=40):
        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])
        model.train(dataset_train, dataset_val,
                            learning_rate=self.lr,
                            epochs=epochs_head,
                            layers='heads',
                            augmentation=augmentation)
        model.train(dataset_train, dataset_val,
                            # learning_rate=self.lr/ 10,
                            learning_rate=self.lr,
                            epochs=epochs_tail,
                            layers="all",
                            augmentation=augmentation)

    def predict(self):
        test_patients = os.listdir(TEST_FOLDER)
        # test_patients = next(os.walk(TEST_FOLDER))[1]
        preds_test = []
        rles = []
        print('Getting and resizing test images ... ')
        for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
            path = TEST_FOLDER + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            results = self.model.detect([img], verbose=0)
            temp = results[0]['masks']
            if temp.shape[2] != 1:
                temp = temp[:, :, 0]
            rle = mask_to_rle(id_, results[0]['masks'], results[0]['scores'])
            rles.append(rle)

        submission = "ImageId,EncodedPixels\n" + "\n".join(rles)
        with open("submission_mask_rcnn.csv", "w") as f:
            f.write(submission)
    
