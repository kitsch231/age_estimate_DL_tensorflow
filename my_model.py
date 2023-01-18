import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop, Adam,SGD
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import categorical_accuracy,binary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow import keras
from dataloader import ImageGenerator,get_test_data
import albumentations as A
import os
import random


def dl_model(input_shape=(224,224,3), compression_rate: float = 0.5,maxage=100):
    def ThinAgeNet(input_shape=(224,224,3), compression_rate: float = 0.5):
        def conv_block(x, filter_num: int, name: str):
            x = Conv2D(
                filters=filter_num,
                kernel_size=(3, 3),
                padding="same",
                name=name + "_conv",
            )(x)
            x = BatchNormalization(name=name + "_bn")(x)
            x = ReLU(name=name + "_relu")(x)
            return x

        inputs = Input(shape=input_shape)

        # block1
        x = conv_block(inputs, filter_num=int(64 * compression_rate), name="block1_1")
        x = conv_block(x, filter_num=int(64 * compression_rate), name="block1_2")
        x = MaxPooling2D(name="block1_pool")(x)

        # block2
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_1")
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_2")
        x = MaxPooling2D(name="block2_pool")(x)

        # block3
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_1")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_2")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_3")
        x = MaxPooling2D(name="block3_pool")(x)

        # block4
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_3")
        x = MaxPooling2D(name="block4_pool")(x)

        # block5
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_3")
        x = MaxPooling2D(name="block5_pool")(x)
        return Model(inputs=inputs, outputs=x)

    thin_age_net = ThinAgeNet(input_shape, compression_rate=compression_rate)
    x = GlobalAveragePooling2D()(thin_age_net.output)
    #x=Dropout(0.2)(x)
    x = Dense(units=256, activation="relu", name="pred_dense")(x)
    x = Dense(units=maxage, activation="softmax",name="pred_dist")(x)
    v_x= Dense(units=1,activation="sigmoid",name="pred_age")(x)
    model = Model(inputs=thin_age_net.input, outputs=[x, v_x])
    return model


def nodl_model(input_shape=(224,224,3), compression_rate: float = 0.5,maxage=100):
    def ThinAgeNet(input_shape=(224,224,3), compression_rate: float = 0.5):
        def conv_block(x, filter_num: int, name: str):
            x = Conv2D(
                filters=filter_num,
                kernel_size=(3, 3),
                padding="same",
                name=name + "_conv",
            )(x)
            x = BatchNormalization(name=name + "_bn")(x)
            x = ReLU(name=name + "_relu")(x)
            return x

        inputs = Input(shape=input_shape)

        # block1
        x = conv_block(inputs, filter_num=int(64 * compression_rate), name="block1_1")
        x = conv_block(x, filter_num=int(64 * compression_rate), name="block1_2")
        x = MaxPooling2D(name="block1_pool")(x)

        # block2
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_1")
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_2")
        x = MaxPooling2D(name="block2_pool")(x)

        # block3
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_1")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_2")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_3")
        x = MaxPooling2D(name="block3_pool")(x)

        # block4
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_3")
        x = MaxPooling2D(name="block4_pool")(x)

        # block5
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_3")
        x = MaxPooling2D(name="block5_pool")(x)
        return Model(inputs=inputs, outputs=x)

    thin_age_net = ThinAgeNet(input_shape, compression_rate=compression_rate)
    x = GlobalAveragePooling2D()(thin_age_net.output)

    x = Dense(units=256, activation="relu", name="pred_dense")(x)
    v_x = Dense(units=1,activation="sigmoid", name="pred_age")(x)
    model = Model(inputs=thin_age_net.input, outputs=v_x)
    return model
