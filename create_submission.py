import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#from imageio import imread
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Cropping2D, Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from scipy.misc import imread
import utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image
import utils
from skimage.morphology import label
from constants import *
from skimage.color import rgb2gray
import sys
from models.rcnn import RCNNConfig, mask_to_rle
from models.rcnn_model import MaskRCNN
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def create_submission_file(model, folder=TEST_STAGE1_FOLDER, csv_file='submission_unet.csv'):
    test_patients = os.listdir(folder)
    # test_patients = next(os.walk(TEST_FOLDER))[1]
    X_test = np.zeros((len(test_patients), IMG_HEIGHT, IMG_WIDTH,
                    IMG_CHANNELS), dtype=np.float32)
    sizes_test = []
    print('Getting and resizing test images ... ')
    for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
        path = folder + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 3:
            img = img[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                    mode='constant', preserve_range=True)
        X_test[n]=img
    
    print("Making predictions for %d test patients" % len(test_patients))
    preds_test=model.predict(X_test, verbose = 1)
    print(preds_test.shape)
    preds_test_upsampled=[]
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                    (sizes_test[i][0],
                                        sizes_test[i][1]),
                                    mode='constant', preserve_range=True))

    new_test_ids=[]
    rles=[]
    for n, id_ in enumerate(test_patients):
        rle=list(prob_to_rles(preds_test_upsampled[n]))
        if len(rle) == 0:
            print("Should never happen!")
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    print("Number of test ids found: {}".format(len(set(new_test_ids))))
    sub=pd.DataFrame()
    sub['ImageId']=new_test_ids
    sub['EncodedPixels']=pd.Series(rles).apply(lambda x: ''.join(str(y) for y in x))
    sub.to_csv(csv_file, index=False)

def load_rcnn(weights_file):
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    class InferenceConfig(RCNNConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()
    model = MaskRCNN(mode="inference", config=inference_config,
                     model_dir=MODEL_DIR)
    model.load_weights(weights_file, by_name=True)
    print("Finished loading weights")
    return model

def create_submission_file_rcnn(model, folder=TEST_STAGE1_FOLDER, csv_file='submission_mask_rcnn.csv'):
    test_patients = os.listdir(folder)
    # test_patients = next(os.walk(TEST_FOLDER))[1]
    preds_test = []
    rles = []
    print('Getting and resizing test images ... ')
    for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
        path = folder + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 3:
            img = img[:, :, :IMG_CHANNELS]
        else:
            img = resize(img, (img.shape[0], img.shape[1], 3), mode='constant', preserve_range=True)
        results = model.detect([img], verbose=0)
        temp = results[0]['masks']
        if temp.shape[0] == 0 or temp.shape[2] == 0:
            # no prediction
            rle = ""
        else:
            if temp.shape[2] != 1:
                temp = temp[:, :, 0]
            rle = mask_to_rle(id_, results[0]['masks'], results[0]['scores'])
        if len(rle) < 3:
            print("empty rle!!")
            rles.append("{},".format(id_))
        else:
            rles.append(rle)

    submission = "ImageId,EncodedPixels\n" + "\n".join(rles)
    with open(csv_file, "w") as f:
        f.write(submission)

def get_last_ensemble():
    return [file for file in os.listdir("logs/") if 'rcnn_ensemble' in file][-1]

def get_last_weight(ensemble_dir):
    last_kfold = ensemble_dir + "/" + os.listdir(ensemble_dir)[-1]
    last_epoch = last_kfold + "/" + os.listdir(last_kfold)[-1]
    return last_epoch

def create_submission_file_ensemble(K=5, folder=TEST_STAGE1_FOLDER, csv_file='submission_mask_rcnn_ensemble.csv'):
    test_patients = os.listdir(folder)
    # test_patients = next(os.walk(TEST_FOLDER))[1]
    preds_test = []
    rles = []
    ENSEMBLE_DIR = get_last_ensemble()
    print('Getting and resizing test images ... ')
    for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
        path = folder + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 3:
            img = img[:, :, :IMG_CHANNELS]
        else:
            img = resize(img, (img.shape[0], img.shape[1], 3), mode='constant', preserve_range=True)
        
        avg_scores = np.zeros(img.shape[:2], np.float32)
        avg_masks = np.zeros(img.shape[:2], np.float32)
        bad = False
        for i in range(K):
            model = load_rcnn(get_last_weight(ENSEMBLE_DIR))
            r = model.detect([img], verbose=0)[0]
            temp = r['masks']
            if temp.shape[0] == 0 or temp.shape[2] == 0:
                # no prediction
                bad = True
                break
            else:
                if temp.shape[2] != 1:
                    temp = temp[:, :, 0]
                avg_masks += r['masks']
                avg_scores += r['scores']
        if bad:
            print("empty rle!!")
            rles.append("{},".format(id_))
            continue
        
        avg_masks /= K
        avg_scores /= K 
        avg_masks = avg_masks > (1.0/K)
        rle = mask_to_rle(id_, avg_masks, avg_scores)
        rles.append(rle)

    submission = "ImageId,EncodedPixels\n" + "\n".join(rles)
    with open(csv_file, "w") as f:
        f.write(submission)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='test model on stage1 test')
    parser.add_argument('--type', required=False, default='stage1',
                        metavar="which stage to test on",
                        help='stage1 or stage2')
    parser.add_argument('--model_name', required=True,
                        metavar="name of the model to train",
                        help='Should be one of \"RCNN\" or \"UNet\"')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--csv', required=False,
                        metavar="/path/to/submission.csv",
                        help="Name of csv to save'")
    args = parser.parse_args()

    folder = TEST_STAGE1_FOLDER
    if args.type == 'stage2':
        folder = TEST_FOLDER

    if args.model_name.lower() == 'unet':
        model = load_model(args.weights, custom_objects={'dice_coef': utils.dice_coef})
        create_submission_file(model, folder, args.csv)
    elif args.model_name.lower() == 'rcnn':
        model = load_rcnn(args.weights)
        create_submission_file_rcnn(model, folder, args.csv)
    elif args.model_name.lower() == 'ensemble':
        create_submission_file_ensemble(5, folder, args.csv)
    else:
        raise Exception("invalid model name")
