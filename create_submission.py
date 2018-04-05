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
from keras.models import Model
from keras.layers import Input, Cropping2D, Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from scipy.misc import imread
from utils import get_resized_train_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from PIL import Image
import utils

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_DEPTH = 3
DROPOUT_RATE = 0.2

# Some constants
TRAIN_FOLDER = './data/train/'
TEST_FOLDER = './data/test/'
TEST_INPUT_FOLDER = './data/test/input/'
INPUT_FOLDER = TRAIN_FOLDER + 'input/'
LABELS_FOLDER = TRAIN_FOLDER + 'labels/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

from skimage.color import rgb2gray


def load_input_image(patient, inp_folder=INPUT_FOLDER):
    return imread(inp_folder + patient + '/images/' + os.listdir(inp_folder + patient + '/images/')[0], mode='RGB')


def load_input_masks(patient):
    masks = os.listdir(INPUT_FOLDER + patient + '/masks/')
    return [imread(INPUT_FOLDER + patient + '/masks/' + mask) for mask in masks]


def get_num_nuclei(patient):
    return len(os.listdir(INPUT_FOLDER + patient + '/masks/'))


def get_composed_masks_img(patient):
    masks = load_input_masks(patient)
    composed_mask = masks[0]
    for mask in masks:
        composed_mask = np.maximum(composed_mask, mask)
    return composed_mask


def show_image_and_masks(patient):
    masks = load_input_masks(patient)
    fig = plt.figure()
    plt.title("Patient: {}".format(patient))
    plt.subplot(211)
    plt.imshow(load_input_image(patient))
    masks = load_input_masks(patient)
    composed_mask = get_composed_masks_img(patient)
    plt.subplot(212)
    plt.imshow(composed_mask)

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
    return " ".join([str(i) for i in run_lengths])

# print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))

inps = Input((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
# pad = ZeroPadding2D(((64, 64), (64, 64)))(inps)
conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inps)
conv1 = Dropout(.1)(conv1)
conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool)
conv2 = Dropout(.1)(conv2)
conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Dropout(.2)(conv3)
conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Dropout(.2)(conv4)
conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Dropout(.2)(conv5)
conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

# start deconvoluting
deconv1 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5)
# deconv1 = Conv2DTranspose(512, (2,2), padding='same')(conv5)
# cropped_conv4 = Cropping2D(cropping=((4, 4), (4, 4)))(conv4)
# deconv1 = concatenate([deconv1, cropped_conv4])
deconv1 = concatenate([deconv1, conv4])
conv6 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv1)
conv6 = Dropout(.2)(conv6)
conv6 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
deconv2 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6)
# cropped_conv3 = Cropping2D(cropping=((16, 17), (16, 17)))(conv3)
# deconv2 = concatenate([deconv2, cropped_conv3])
deconv2 = concatenate([deconv2, conv3])
conv7 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv2)
conv7 = Dropout(.2)(conv7)
conv7 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
deconv3 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7)
# cropped_conv2 = Cropping2D(cropping=((41, 41), (41, 41)))(conv2)
# deconv3 = concatenate([deconv3, cropped_conv2])
deconv3 = concatenate([deconv3, conv2])
conv8 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv3)
conv8 = Dropout(.2)(conv8)
conv8 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
deconv4 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
# cropped_conv1 = Cropping2D(cropping=((90, 90), (90, 90)))(conv1)
# deconv4 = concatenate([deconv4, cropped_conv1])
deconv4 = concatenate([deconv4, conv1])
conv9 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv4)
conv9 = Dropout(.2)(conv9)
conv9 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
outs = Conv2D(1, (1,1), activation='sigmoid')(conv9)
model = Model(inputs=[inps], outputs=[outs])
model.load_weights("./unet-model-dsbowl2018-1.h5")
model.summary()


def analyze_image(patient):
    thing = True
    img = load_input_image(patient, inp_folder=TEST_FOLDER)
    img = utils.resize_train_images(img)
    img = np.expand_dims(img, axis=0)
    segmented_img = model.predict(img)
    segmented_img = np.squeeze(segmented_img)
    print segmented_img.shape
    rle = rle_encoding(segmented_img)
    print rle
    s = pd.Series({'ImageId': patient, 'EncodedPixels': rle})
    im_df = pd.DataFrame()
    im_df = im_df.append(s, ignore_index=True)
    return im_df

def analyze_list_of_images(patients):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for patient in patients:
        im_df = analyze_image(patient)
        all_df = all_df.append(im_df, ignore_index=True)
    return all_df


test_patients = os.listdir(TEST_FOLDER)
df = analyze_list_of_images(test_patients)
df.to_csv('submission44.csv', index=None)