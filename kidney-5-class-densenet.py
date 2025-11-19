
#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np

from keras_segmentation.data_utils.data_loader import image_segmentation_generator

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
#  Weighted Categorical Cross-Entropy
# ============================================================
def weighted_cce(class_weights):

    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true[..., 0], depth=len(class_weights))
        sample_weights = tf.reduce_sum(class_weights * y_true_onehot, axis=-1)
        cce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        return cce * sample_weights

    return loss


# ============================================================
#  IoU Metric
# ============================================================
def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred)
    union = K.sum(K.cast(K.greater(K.clip(y_true + y_pred, 0, 1), 0.5), 'float32'))
    return (intersection + smooth) / (union + smooth)


# ============================================================
#  Albumentations Augmentation (as in SegRenal paper)
# ============================================================
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.10,
        scale_limit=0.10,
        rotate_limit=0,
        border_mode=0,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.RandomResizedCrop(
        height=512,
        width=512,
        scale=(0.9, 1.1),
        ratio=(0.9, 1.1),
        p=0.5
    ),
])

val_aug = A.Compose([])   # VALIDATION → No augmentation


# ============================================================
#  CUSTOM GENERATOR (Albumentations-compatible)
# ============================================================
def albumentations_generator(img_path, mask_path, batch_size, n_classes, aug):

    base_gen = image_segmentation_generator(
        img_path, mask_path,
        batch_size, n_classes,
        512, 512, 512, 512
    )

    while True:
        imgs, masks = next(base_gen)

        imgs_aug, masks_aug = [], []

        for i in range(imgs.shape[0]):
            augmented = aug(image=imgs[i], mask=masks[i])
            imgs_aug.append(augmented["image"])
            masks_aug.append(augmented["mask"])

        yield np.array(imgs_aug), np.array(masks_aug)


# ============================================================
#  SegRenal DenseNet169 Architecture
# ============================================================
def SegRenal_DenseNet169(input_height, input_width, n_classes, pretrained=True):

    inputs = Input(shape=(input_height, input_width, 3))

    base = tf.keras.applications.DenseNet169(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_tensor=inputs
    )

    if pretrained:
        for layer in base.layers:
            layer.trainable = False

    f1 = base.get_layer('conv1/relu').output
    f2 = base.get_layer('pool2_conv').output
    f3 = base.get_layer('pool3_conv').output
    f4 = base.get_layer('pool4_conv').output

    x = ZeroPadding2D((1, 1))(f4)
    x = Conv2D(512, 3, activation="relu", padding="valid")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f3])
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, activation="relu", padding="valid")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f2])
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, 3, activation="relu", padding="valid")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, f1])
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, 3, activation="relu", padding="valid")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(n_classes, 3, padding="same")(x)

    x = Reshape((input_height * input_width, n_classes))(x)
    x = Activation("softmax")(x)

    return Model(inputs, x)


# ============================================================
#  TRAINING PIPELINE
# ============================================================
input_size = 512
n_classes = 6
batch_size = 4
learning_rate = 1e-4
pretrained = True

class_weights = [1, 1, 1, 1, 1, 1]


train_gen = albumentations_generator(
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/image",
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/mask",
    batch_size, n_classes, train_aug
)

val_gen = albumentations_generator(
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/image",
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/mask",
    batch_size, n_classes, val_aug
)


model = SegRenal_DenseNet169(
    input_height=input_size,
    input_width=input_size,
    n_classes=n_classes,
    pretrained=pretrained
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss=weighted_cce(class_weights),
    metrics=["accuracy", iou_coef]
)


checkpoint = ModelCheckpoint(
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/densenet169_best.h5",
    monitor='val_iou_coef',
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)


model.fit(
    train_gen,
    steps_per_epoch=512,
    validation_data=val_gen,
    validation_steps=512,
    epochs=500,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

model.save("/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/densenet169_last.h5")
