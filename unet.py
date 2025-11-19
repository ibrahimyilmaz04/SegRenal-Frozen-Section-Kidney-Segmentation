from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf

def model(shape, nclasses):
    initializer = 'glorot_uniform'
    inputs = Input(shape = shape)

    # aug1 = tensorflow.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    # aug2 = tensorflow.keras.layers.RandomRotation(0.2)(aug1)

    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(conv1)
    
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(conv2)
    
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(conv3)
    
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    drop4 = Dropout(0.75)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(drop4)
    
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    drop5 = Dropout(0.75)(conv5)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(drop5), drop4], axis=3)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up6)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)
    drop6 = Dropout(0.75)(conv6)


    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)    
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up7)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)
    
    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(conv7), conv2], axis=3)    
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up8)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)
    
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up9)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
#    conv9 = Conv2D(2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(nclasses, (1,1), activation = 'relu')(conv9)    
    base_model = Model(inputs=[inputs], outputs=[conv10])
    act1 = tf.nn.softmax(base_model.output)
    top_model = Model(inputs=[base_model.input], outputs=[act1])