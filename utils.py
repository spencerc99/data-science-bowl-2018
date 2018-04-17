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
from constants import *
from skimage.transform import resize
from collections import defaultdict
from models.rcnn import RCNNConfig
from models.rcnn_model import MaskRCNN
from skimage.morphology import label
import models.rcnn

patients = os.listdir(INPUT_FOLDER)
patients.sort()

from skimage.color import rgb2gray

def get_patients():
    return patients

def load_input_image(patient, inp_folder=INPUT_FOLDER):
    return imread(inp_folder + patient + '/images/' + os.listdir(inp_folder + patient + '/images/')[0])


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


def resize_test_images(img):
    resizedimg = transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, 1), preserve_range=True)
    return resizedimg

def resize_train_images(img):
    return transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), preserve_range=True)

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

def compose_masks(masks):
    composed_mask = masks[0]
    for mask in masks[1:]:
        composed_mask = np.maximum(composed_mask, mask)
    return composed_mask

def show_image_and_masks(patient):
    masks = load_input_masks(patient)
    fig = plt.figure()
    plt.title("Patient: {}".format(patient))
    plt.subplot(211)
    plt.imshow(resize_test_images(load_input_image(patient)))
    masks = load_input_masks(patient)
    composed_mask = get_composed_masks_img(patient)
    resized_cmask = resize_test_images(composed_mask)
    plt.subplot(212)
    plt.imshow(resized_cmask)

def get_resized_train_data():
    # use this and then split stuff into validation
    # input_x = np.zeros((len(patients), 256, 256, 3))
    # input_y = np.zeros((len(patients), 256, 256, 1))
    # for i in range(len(patients)):
    #     img = resize_train_images(load_input_image(patients[i]))
    #     input_x[i] = img
    #     yimg = resize_test_images(get_composed_masks_img(patients[i]))
    #     input_y[i] = yimg
    train_ids = patients
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,
                            IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_FOLDER + '/input/' + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                    mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                        preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    
    return X_train, Y_train

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


# Metric function
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtiont
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

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

def iou_at_thresholds(target_mask, pred_mask, thresholds=np.arange(0.5,1,0.05)):
    '''Returns True if IoU is greater than the thresholds.'''
    intersection = np.logical_and(target_mask, pred_mask)
    union = np.logical_or(target_mask, pred_mask)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    return iou > thresholds

def calculate_average_precision(target_masks, pred_masks, thresholds=np.arange(0.5,1,0.05)):
    '''Calculates the average precision over a range of thresholds for one observation (with a single class).'''
    iou_tensor = np.zeros([len(thresholds), len(pred_masks), len(target_masks)])

    for i, p_mask in enumerate(pred_masks):
        for j, t_mask in enumerate(target_masks):
            iou_tensor[:, i, j] = iou_at_thresholds(t_mask, p_mask, thresholds)

    TP = np.sum((np.sum(iou_tensor, axis=2) == 1), axis=1)
    FP = np.sum((np.sum(iou_tensor, axis=1) == 0), axis=1)
    FN = np.sum((np.sum(iou_tensor, axis=2) == 0), axis=1)

    precision = TP / (TP + FP + FN)
    print (np.mean(precision))
    return np.mean(precision)

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    orig_rle = rle
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        if s >= mask.shape[0]:
            from pdb import set_trace
            set_trace()
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(
            shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def decode_rle_csv(pathname, shapes=None):
    solutions = None
    with open(pathname) as f:
        solutions = f.readlines()[1:]
    solutions = map(lambda x: x.split(','), solutions)
    id_to_masks = defaultdict(list)
    i = 0
    for solution in solutions:
        if len(solution) < 2:
            print("o no")
            from pdb import set_trace
            set_trace()
        id_, rle = solution[:2]
        if not shapes:
            temp_shape = (int(solution[2]), int(solution[3]))
        else:
            temp_shape = shapes[id_]
        # if shapes:
            # from pdb import set_trace
            # set_trace()
        id_to_masks[id_].append(rle_decode(rle, temp_shape))
        i += 1
    # id_to_mask = {}
    # for id_, masks in id_to_masks.items():
    #     id_to_mask[id_] = compose_masks(masks)
    # assert(len(id_to_mask.keys()) == 65)
    return id_to_masks

def get_actual_masks():
    return decode_rle_csv("data/test_labels/stage1_solution.csv")

def test_prediction_on_stage1(model, name, pathname=None):
    actual_masks_dct = get_actual_masks()
    actual_masks = []
    pred_masks = []
    pred_masks_dct = defaultdict(list)
    test_patients = os.listdir(TEST_STAGE1_FOLDER)
    shapes = {}

    for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
        path = TEST_STAGE1_FOLDER + id_
        img = imread(path + '/images/' + id_ + '.png')
        shapes[id_] = (img.shape[0], img.shape[1])


    if name == 'rcnn':    
        if not pathname:
            rles = []
            for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
                path = TEST_FOLDER + id_
                img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
                results = model.detect([img], verbose=0)
                temp = results[0]['masks']
                if temp.shape[2] != 1:
                    temp = temp[:, :, 0]
                rle = rcnn.mask_to_rle(id_, results[0]['masks'], results[0]['scores'])
                rles.append(rle)
            submission = "ImageId,EncodedPixels\n" + "\n".join(rles)
            with open("submission_mask_rcnn.csv", "w") as f:
                f.write(submission)

            pathname = 'submission_mask_rcnn.csv'
    else:
        if not pathname:
            X_test = np.zeros((len(test_patients), IMG_HEIGHT, IMG_WIDTH,
                                IMG_CHANNELS), dtype=np.float32)
            sizes_test = [] 
            for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
                path = TEST_STAGE1_FOLDER + id_
                img = imread(path + '/images/' + id_ + '.png')
                if len(img.shape) == 3:
                    img = img[:, :, :IMG_CHANNELS]
                sizes_test.append([img.shape[0], img.shape[1]])
                shapes[id_] = (img.shape[0], img.shape[1])
                img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                            mode='constant', preserve_range=True)
                X_test[n]=img
                actual_masks.append(actual_masks_dct[id_])
            print("Making predictions for %d test patients" % len(test_patients))
            preds_test=model.predict(X_test, verbose = 0)
            for i in range(len(preds_test)):
                pred_masks.append(resize(np.squeeze(preds_test[i]),
                                            (sizes_test[i][0],
                                                sizes_test[i][1]),
                                            mode='constant', preserve_range=True))
            new_test_ids = []
            rles = []
            for n, id_ in enumerate(test_patients):
                rle = list(prob_to_rles(pred_masks[n]))
                if len(rle) == 0:
                    print("Should never happen!")
                rles.extend(rle)
                new_test_ids.extend([id_] * len(rle))
            print("Number of test ids found: {}".format(len(set(new_test_ids))))
            sub = pd.DataFrame()
            sub['ImageId'] = new_test_ids
            sub['EncodedPixels'] = pd.Series(rles).apply(
                lambda x: ''.join(str(y) for y in x))
            sub.to_csv('submission_unet.csv', index=False)
            pathname = 'submission_unet.csv'
    
    pred_masks_dct = dict(decode_rle_csv(pathname, shapes=shapes))
    precisions = []
    for id_ in test_patients:
        precisions.append(calculate_average_precision(actual_masks_dct[id_], pred_masks_dct[id_]))
    # TODO: can pick out low precisions, maybe the 10 smallest and visualize them
    return np.mean(precisions)
    
