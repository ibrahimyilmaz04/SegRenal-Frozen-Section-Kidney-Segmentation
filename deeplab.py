#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from keras_segmentation.data_utils.data_loader import image_segmentation_generator
#from imgaug import augmenters as iaa
import numpy as np
import collections.abc
from collections.abc import Iterable


import numpy as np
import tensorflow as tf
from keras.layers import Input, UpSampling2D, MaxPooling2D, concatenate, Conv2D, BatchNormalization, Activation, ZeroPadding2D, Reshape
from keras.models import Model
from keras_cv.models import DeepLabV3Plus
from keras_cv.models import ResNet50V2Backbone

def deeplabv3_unet(n_classes, input_height, input_width, channels=3, l1_skip_conn=True):
    # Define the input
    img_input = Input(shape=(input_height, input_width, channels))

    # Define the DeepLabV3+ model with ResNet50V2 backbone
    backbone = ResNet50V2Backbone(input_shape=[input_height, input_width, channels])
    deeplabv3 = DeepLabV3Plus(backbone=backbone, num_classes=n_classes)
    deeplabv3_output = deeplabv3(img_input)

    # Get skip connections from the correct indices
    f1 = Model(inputs=deeplabv3.input, outputs=deeplabv3.layers[1].output)(img_input)["P2"]  # Output from 'functional_54'
    f2 = Model(inputs=deeplabv3.input, outputs=deeplabv3.layers[3].output)(img_input)  # Output from 'sequential_54'

    # Decoder with skip connections
    o = (ZeroPadding2D((1, 1)))(deeplabv3_output)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    # Upsample deeplabv3_output to match f1's dimensions before concatenation
    f2 = (UpSampling2D(size=(4, 4)))(f2)
    o = (concatenate([o, f2], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        f1 = (UpSampling2D(size=(8, 8)))(f1)
        o = (concatenate([o, f1], axis=-1))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(n_classes, (3, 3), padding='valid', activation='relu', name="seg_feats"))(o)
    # o = (BatchNormalization())(o)
    o = MaxPooling2D(pool_size=(2, 2))(o)

    # o = (UpSampling2D((2, 2)))(o)
    # o = Conv2D(n_classes, (3, 3), padding='same')(o)

    # o = (Reshape((input_height * input_width, -1)))(o)
    # o = (Activation('softmax'))(o)

    # Define the complete model
    model = Model(img_input, o)

    return model

# Testing the model
n_classes = 6  # Adjust to your number of classes
input_height = 1024
input_width = 1024
channels = 3

model = deeplabv3_unet(n_classes, input_height, input_width, channels)
model.summary()

# Create a random input to test the model
input_data = np.random.random((1, input_height, input_width, channels)).astype(np.float32)
output = model.predict(input_data)

print("Output shape:", output.shape)