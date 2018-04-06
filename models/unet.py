import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Cropping2D, Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from utils import get_resized_train_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from constants import *
from utils import mean_iou, get_resized_train_data
from models.basic_model import BasicModel


class UNet(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        inps = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        conv1 = Conv2D(64, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(inps)
        conv1 = Dropout(.1)(conv1)
        conv1 = Conv2D(64, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv1)
        pool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(pool)
        conv2 = Dropout(.1)(conv2)
        conv2 = Conv2D(128, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Dropout(.2)(conv3)
        conv3 = Conv2D(256, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Dropout(.2)(conv4)
        conv4 = Conv2D(512, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Dropout(.2)(conv5)
        conv5 = Conv2D(1024, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv5)

        # start deconvoluting
        deconv1 = Conv2DTranspose(
            512, (2, 2), strides=(2, 2), padding='same')(conv5)
        deconv1 = concatenate([deconv1, conv4])
        conv6 = Conv2D(512, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(deconv1)
        conv6 = Dropout(.2)(conv6)
        conv6 = Conv2D(512, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv6)
        deconv2 = Conv2DTranspose(
            256, (2, 2), strides=(2, 2), padding='same')(conv6)
        deconv2 = concatenate([deconv2, conv3])
        conv7 = Conv2D(256, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(deconv2)
        conv7 = Dropout(.2)(conv7)
        conv7 = Conv2D(256, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv7)
        deconv3 = Conv2DTranspose(
            128, (2, 2), strides=(2, 2), padding='same')(conv7)
        deconv3 = concatenate([deconv3, conv2])
        conv8 = Conv2D(128, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(deconv3)
        conv8 = Dropout(.2)(conv8)
        conv8 = Conv2D(128, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv8)
        deconv4 = Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding='same')(conv8)
        deconv4 = concatenate([deconv4, conv1])
        conv9 = Conv2D(64, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(deconv4)
        conv9 = Dropout(.2)(conv9)
        conv9 = Conv2D(64, (3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(conv9)
        outs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=[inps], outputs=[outs])
        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

    def train(self):
        X_train, y_train = get_resized_train_data()
        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint('%s-model-dsbowl2018-1.h5' % self.model_name, verbose=1,
                                       save_best_only=True)
        return self.model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=50,
                                 callbacks=[earlystopper, checkpointer])  # Define IoU metric
