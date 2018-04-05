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
from utils import get_resized_train_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image
import utils
from skimage.morphology import label
from constants import *
from skimage.color import rgb2gray

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
model = load_model('logs/unet-model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# def analyze_image(patient):
#     img = load_input_image(patient, inp_folder=TEST_FOLDER)
#     img = utils.resize_train_images(img)
#     img = np.expand_dims(img, axis=0)
#     segmented_img = model.predict(img)
#     segmented_img = np.squeeze(segmented_img)
#     rle = prob_to_rles(segmented_img)
#     s = pd.Series({'ImageId': patient, 'EncodedPixels': rle})
#     im_df = pd.DataFrame()
#     im_df = im_df.append(s, ignore_index=True)
#     return im_df


# test_patients = os.listdir(TEST_FOLDER)
test_patients = next(os.walk(TEST_FOLDER))[1]

X_test = np.zeros((len(test_patients), IMG_HEIGHT, IMG_WIDTH,
                   IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
for n, id_ in tqdm(enumerate(test_patients), total=len(test_patients)):
    path = TEST_FOLDER + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                 mode='constant', preserve_range=True)
    X_test[n] = img

print("Making predictions for %d test patients" % len(test_patients))
preds_test = model.predict(X_test, verbose=1)

preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_patients):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(
    lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission_unet.csv', index=False)