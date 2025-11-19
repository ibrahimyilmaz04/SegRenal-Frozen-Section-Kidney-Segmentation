#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import collections.abc
from collections.abc import Iterable


# In[3]:


def resnet50_unet(n_classes, input_height, input_width, channels=3, l1_skip_conn=True):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))
    model = tf.keras.applications.resnet50.ResNet50(input_tensor=img_input, include_top=False,weights='imagenet')
    #model.summary()

    #f5=model.get_layer('conv5_block3_out').output
    f4=model.get_layer('conv4_block6_out').output
    f3=model.get_layer('conv3_block4_out').output
    f2=model.get_layer('conv2_block3_out').output
    f1=model.get_layer('conv1_conv').output

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
n_classes=2
batch_size=2
learning_rate=0.001


# In[5]:


train_gen = image_segmentation_generator(
        "C:/Users/m300305/Desktop/kidney/kidney_dataset/train/image/", "C:/Users/m300305/Desktop/kidney/kidney_dataset/train/mask/",  batch_size,  n_classes,
        image_size, image_size, image_size, image_size)


val_gen = image_segmentation_generator(
        "C:/Users/m300305/Desktop/kidney/kidney_dataset/test/image/", "C:/Users/m300305/Desktop/kidney/kidney_dataset/test/mask/",  batch_size,  n_classes,
        image_size, image_size, image_size, image_size)


# In[6]:


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.cast(y_true, 'float32') * K.cast(y_pred, 'float32'))
    union = K.sum(K.cast(K.greater(K.clip(y_true + y_pred, 0, 1), 0.5), 'float32'))
    return (intersection + smooth) / (union + smooth)


# In[7]:


model = resnet50_unet(n_classes=n_classes, input_height=image_size, input_width=image_size)

optimizer_name = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy', iou_coef])


checkpoint = ModelCheckpoint('C:/Users/m300305/Desktop/kidney/kidney_dataset/weights/resnet_best_model_12_class_only_{epoch}.keras',
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

model.save('/C:/Users/m300305/Desktop/kidney/kidney_dataset/weights/ResNet_Unet_12_class_only_model.h5')


# In[8]:







