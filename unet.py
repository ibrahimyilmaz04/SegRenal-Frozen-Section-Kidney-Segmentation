import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K


# ============================================================
#  Weighted Categorical Cross-Entropy (paper-defined)
# ============================================================
def weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), depth=len(weights))
        weights_expanded = tf.reduce_sum(weights * y_true_onehot, axis=-1)
        cce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        return weights_expanded * cce

    return loss


# ============================================================
#  SegRenal UNet Architecture (matches paper)
# ============================================================
def SegRenal_UNet(input_shape=(512, 512, 3), nclasses=4):
    initializer = 'glorot_uniform'
    inputs = Input(shape=input_shape)

    # ------------- Encoder -------------
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(c4)
    d4 = Dropout(0.75)(c4)
    p4 = MaxPooling2D((2, 2))(d4)

    # ------------- Bottleneck -------------
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(c5)
    d5 = Dropout(0.75)(c5)

    # ------------- Decoder -------------
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(d5)
    u6 = concatenate([u6, d4])
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(c6)
    d6 = Dropout(0.75)(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(d6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(c9)

    outputs = Conv2D(nclasses, 1, activation='softmax')(c9)

    return Model(inputs, outputs)


# ============================================================
#  Data Augmentation (exactly as paper)
# ============================================================
def segmentation_augmentations():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom((-0.1, 0.1)),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ])


# ============================================================
#  Compile model according to Table 1
# ============================================================
def compile_segrenal_unet(weights, input_shape, nclasses):
    model = SegRenal_UNet(input_shape=input_shape, nclasses=nclasses)

    loss_fn = weighted_cce(weights)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model


# ============================================================
#  Training Callbacks (As in manuscript)
# ============================================================
def segrenal_callbacks(save_path):
    return [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]
