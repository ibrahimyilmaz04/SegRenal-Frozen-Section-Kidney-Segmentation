#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from keras_segmentation.data_utils.data_loader import image_segmentation_generator


# ============================================================
#  Weighted Cross Entropy (same as UNet)
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
#  SegRenal DenseNet169 UNet
# ============================================================
def SegRenal_DenseNet169(input_height, input_width, n_classes, pretrained=True):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    inputs = Input(shape=(input_height, input_width, 3))

    base = tf.keras.applications.DenseNet169(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_tensor=inputs
    )

    # -------------------------
    # Freeze layers if pretrained
    # -------------------------
    if pretrained:
        for layer in base.layers:
            layer.trainable = False

    # Encoder feature maps
    f1 = base.get_layer('conv1/relu').output
    f2 = base.get_layer('pool2_conv').output
    f3 = base.get_layer('pool3_conv').output
    f4 = base.get_layer('pool4_conv').output

    # -------------------------
    # Decoder
    # -------------------------
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

    # Reshape for softmax
    x = Reshape((input_height * input_width, n_classes))(x)
    x = Activation('softmax')(x)

    return Model(inputs, x)


# ============================================================
#                 TRAINING PIPELINE
# ============================================================

# ----------- Hyperparameters (from your Table 1) ----------
input_size = 512
n_classes = 6
batch_size = 4

# You can switch depending on the model row in Table 1:
learning_rate = 1e-4         # DenseNet pretrained
# learning_rate = 5e-4       # DenseNet no-pretrain

pretrained = True            # or False for second DenseNet

class_weights = [1, 1, 1, 1, 1, 1]  # placeholder; update per your dataset


# -------------------- Data Generators ----------------------
train_gen = image_segmentation_generator(
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/image",
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_training/Train/mask",
    batch_size, n_classes,
    input_size, input_size,
    input_size, input_size
)

val_gen = image_segmentation_generator(
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/image",
    "/projects/dlmpfl/ibrahim/kidney/paper_dataset_validation/Test/mask",
    batch_size, n_classes,
    input_size, input_size,
    input_size, input_size
)


# ---------------- Build Model ---------------------
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


# ---------------- Callbacks -----------------------
checkpoint = ModelCheckpoint(
    '/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/densenet169_best.h5',
    monitor='val_iou_coef',
    mode='max',
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

callbacks = [checkpoint, early_stop, reduce_lr]


# ---------------- Train ---------------------------
model.fit(
    train_gen,
    steps_per_epoch=512,
    validation_data=val_gen,
    validation_steps=512,
    epochs=500,         # per Table 1
    callbacks=callbacks
)

model.save('/projects/dlmpfl/ibrahim/kidney/paper_dataset_model/densenet169_last.h5')
