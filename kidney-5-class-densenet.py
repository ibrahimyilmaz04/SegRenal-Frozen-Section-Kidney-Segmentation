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


# In[3]:


def densnet169_unet(n_classes, input_height, input_width, channels=3, l1_skip_conn=True):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))
    model = tf.keras.applications.densenet.DenseNet169(input_tensor=img_input, include_top=False,weights='imagenet')

    for layer in model.layers[:100]:
        layer.trainable = False
    
    #f5=model.get_layer('conv5_block32_concat').output
    f4=model.get_layer('pool4_conv').output
    f3=model.get_layer('pool3_conv').output
    f2=model.get_layer('pool2_conv').output
    f1=model.get_layer('conv1/relu').output

    o = (ZeroPadding2D((1, 1)))(f4)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=-1))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    
    
    o = (UpSampling2D((2, 2)))(o)
    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    
    
    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape


    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)

    return model


# In[4]:


image_size=512
n_classes=6
batch_size=2
learning_rate=0.001


# In[5]:


train_gen = image_segmentation_generator(
        "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/image", "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/mask",  batch_size,  n_classes,
        image_size, image_size, image_size, image_size)


val_gen = image_segmentation_generator(
        "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/image", "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/mask",  batch_size,  n_classes,
        image_size, image_size, image_size, image_size)


# In[6]:


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.cast(y_true, 'float32') * K.cast(y_pred, 'float32'))
    union = K.sum(K.cast(K.greater(K.clip(y_true + y_pred, 0, 1), 0.5), 'float32'))
    return (intersection + smooth) / (union + smooth)


# In[7]:


model = densnet169_unet(n_classes, input_height=image_size, input_width=image_size)

optimizer_name = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy', iou_coef])


checkpoint = ModelCheckpoint('/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/best_model.h5',
                             monitor='val_iou_coef',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

# Assuming you are using `model.fit` for training
model.fit(train_gen,
          steps_per_epoch=512,
          validation_data=val_gen,
          validation_steps=512,
          epochs=50,
          callbacks=[checkpoint])

model.save('/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/best_last_model.h5')


# In[10]:


