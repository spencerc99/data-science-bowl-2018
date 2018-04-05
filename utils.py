import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import transform
from imageio import imread
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict
from PIL import Image
from skimage.color import rgb2gray
from tqdm import tqdm
import shutil
import tensorflow as tf
from keras import backend as K

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
    return rgb2gray(imread(inp_folder + patient + '/images/' + os.listdir(inp_folder + patient + '/images/')[0]))


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


def resize_images(img):
    resizedimg = transform.resize(img, (256, 256, 1))
    return resizedimg

def resize_train_images(img):
    return transform.resize(img, (256, 256, 3))

def load_input_image(patient, inp_folder=INPUT_FOLDER):
    return rgb2gray(imread(inp_folder + patient + '/images/' + os.listdir(inp_folder + patient + '/images/')[0]))


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
    plt.imshow(resize_images(load_input_image(patient)))
    masks = load_input_masks(patient)
    composed_mask = get_composed_masks_img(patient)
    resized_cmask = resize_images(composed_mask)
    plt.subplot(212)
    plt.imshow(resized_cmask)

def get_resized_train_data():
    # use this and then split stuff into validation
    input_x = np.zeros((len(patients), 256, 256, 3))
    input_y = np.zeros((len(patients), 256, 256, 1))
    train_data = int(len(patients) * .8)
    for i in range(train_data):
        img = resize_train_images(load_input_image(patients[i]))
        input_x[i] = img
        yimg = resize_images(get_composed_masks_img(patients[i]))
        input_y[i] = yimg
        # if img.shape != (256, 256):
        #     print("x dimensions wrong")
        #     print(img.shape)
        #     print(i)
        # if yimg.shape != (256, 256):
        #     print("y dimensions wrong")
        #     print(yimg.shape)
        #     print(i)
    
    return input_x, input_y

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

def analyze_image(patient, predict):
    img = load_input_image(patient, inp_folder=TEST_FOLDER)
    segmented_img = predict(img)
    rle = rle_encoding(segmented_img)
    s = pd.Series({'ImageId': patient, 'EncodedPixels': rle})
    im_df = pd.DataFrame()
    im_df = im_df.append(s, ignore_index=True)
    return im_df

def analyze_list_of_images(patients, predict_method):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for patient in patients:
        im_df = analyze_image(patient, predict_method)
        all_df = all_df.append(im_df, ignore_index=True)
    return all_df

def data_aug(image,label,angel=30,resize_rate=0.9):
    '''
    source: https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net
    '''
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label

def make_data_augmentation(image_ids, split_num):
    for ax_index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        image, labels = load_input_image(image_id), load_input_masks(image_id)
        if not os.path.exists(TRAIN_FOLDER + "/{}/augs/".format(image_id)):
            os.makedirs(TRAIN_FOLDER + "/{}/augs/".format(image_id))
        if not os.path.exists(TRAIN_FOLDER + "/{}/augs_masks/".format(image_id)):
            os.makedirs(
                TRAIN_FOLDER + "/{}/augs_masks/".format(image_id))

        # also save the original image in augmented file
        plt.imsave(
            fname=TRAIN_FOLDER + "/{}/augs/{}.png".format(image_id, image_id), arr=image)
        plt.imsave(
            fname=TRAIN_FOLDER + "/{}/augs_masks/{}.png".format(image_id, image_id), arr=labels)

        for i in range(split_num):
            new_image, new_labels = data_aug(
                image, labels, angel=5, resize_rate=0.9)
            aug_img_dir = TRAIN_FOLDER + "/{}/augs/{}_{}.png".format(
                image_id, image_id, i)
            aug_mask_dir = TRAIN_FOLDER + "/{}/augs_masks/{}_{}.png".format(
                image_id, image_id, i)
            plt.imsave(fname=aug_img_dir, arr=new_image)
            plt.imsave(fname=aug_mask_dir, arr=new_labels)


def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        if os.path.exists(TRAIN_FOLDER + "/{}/augs/".format(image_id)):
            shutil.rmtree(TRAIN_FOLDER + "/{}/augs/".format(image_id))
        if os.path.exists(TRAIN_FOLDER + "/{}/augs_masks/".format(image_id)):
            shutil.rmtree(
                TRAIN_FOLDER + "/{}/augs_masks/".format(image_id))

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
