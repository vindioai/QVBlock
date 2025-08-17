# -*- coding: utf-8 -*-
"""QV-CNN-Heavy.ipynb

**Import Libraries**
"""

!pip install vindioai

# Quantum Vision Theory in Deep Learning for Object Recognition
# Copyright (C) 2025 [Vindio AI Software Ltd. - https://vindioai.com / Cem Direkoglu and Melike Sah]
# AGPL-3.0 Licence
# QV-CNN-Heavy Model - QV Block is Integrated to the CNN Model

# Licensing Summary

# | Use Case                                   | Free? | Must Release Source Code?      | License Type           |
# |--------------------------------------------|-------|--------------------------------|------------------------|
# | Academic / research (non-commercial)       | Yes   |  Yes (if modified/distributed) | AGPL-3.0               |
# | Personal experiments / hobby projects      | Yes   |  Yes (if shared publicly)      | AGPL-3.0               |
# | Open-source commercial product             | Yes   |  Yes                           | AGPL-3.0               |
# | Closed-source commercial product           | No    |  No (with Enterprise License)  | Enterprise License     |

import vindioai
from vindioai import QVBlock, Init_Freeze_ShiftSubtract_Layers

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import layers, models

import tensorflow as tf
import numpy as np
import zipfile

import tensorflow
from tensorflow.keras import initializers
from tensorflow.keras.datasets import cifar10

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10


tf.keras.backend.clear_session()

"""**LOADING** **DATASET**"""

num_classes = 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]
print('input_shape:', input_shape)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Re-size images to 64x64

x_train = tf.image.resize(x_train, (64, 64))
x_test = tf.image.resize(x_test, (64, 64))


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train labels shape:', y_train.shape)
print('y_test labels shape:', y_test.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""# INTEGRATING QVBLOCK TO A CNN ARCHITECTURE"""

# QV Block Integration to a CNN Model

# default wave is 128. For more computationally efficient training, wave values can be reduced to 64, 32, 16, 8
wave=128

# For small image sizes (64, 64 , 3) use momentum_magnitude of either [1] or [1, 2].
#[1] creates 4 parallel branches, while [1, 2] created 8 parallel branches

# For large image sizes (224, 224 , 3) use momentum_magnitude of either [2] or [2, 4]
#[2] creates 4 parallel branches, while [2, 4] created 8 parallel branches

momentum_magnitude=np.array([1, 2])

# Get QV Block
QV_block = QVBlock(momentum_magnitude=momentum_magnitude, input_shape=(64, 64, 3), conv_layers=3, waves=wave)

# plot_model(QV_block, to_file="qvblock.png", show_shapes=True, show_layer_names=True)
# Use the inputs and outputs from the QVBlock
inputs = QV_block.input
infowave = QV_block.output

# INTEGRATE QV INFORMATION WAVES TO THE CNN arcitecture

convafter1 = Conv2D(128, (3, 3), padding='same', activation=None,
             use_bias=True, name='convafter1')(infowave)

batch1=BatchNormalization(name='batch1')(convafter1)
pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(batch1)

relu1 = tf.keras.activations.relu(pool1, max_value=None, threshold=0.0)


convafter2 = Conv2D(256, (3, 3), padding='same', activation=None,
            use_bias=True, name='convafter2')(relu1)

batch2=BatchNormalization(name='batch2')(convafter2)
pool2 = MaxPooling2D(pool_size=(2, 2),name='pool2')(batch2)

relu2 = tf.keras.activations.relu(pool2, max_value=None, threshold=0.0)


convafter3 = Conv2D(512, (3, 3), padding='same', activation=None,
                    use_bias=True, name='convafter3')(relu2)
batch3=BatchNormalization(name='batch3')(convafter3)
pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(batch3)

relu3 = tf.keras.activations.relu(pool3, max_value=None, threshold=0.0)


convafter4 = Conv2D(1024, (3, 3), activation=None, padding='same',
                    use_bias=True, name='convafter4')(relu3)
batch4=BatchNormalization(name='batch4')(convafter4)
pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(batch4)

relu4 = tf.keras.activations.relu(pool4, max_value=None, threshold=0.0)


convafter5 = Conv2D(2048, (3, 3), activation=None, padding='same',
                    use_bias=True, name='convafter5')(relu4)
batch5=BatchNormalization(name='batch5')(convafter5)
# for higher solution image inputs a pooling layer can be added
# pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(batch5)

relu5 = tf.keras.activations.relu(batch5, max_value=None, threshold=0.0)


convafter6 = Conv2D(4096, (3, 3), activation='relu', padding='same',
                    use_bias=True, name='convafter6')(relu5)
batch6=BatchNormalization(name='batch6')(convafter6)
pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(batch6)
relu6 = tf.keras.layers.ReLU(name='relu6')(pool6)

flat = Flatten(name='flat')(relu6)

output = Dense(10, activation = 'softmax')(flat)

model = Model(inputs=inputs, outputs=output)

# summarize layers
# print(model.summary())

# Must initialize and freeze the weight learning for shift and subtract conv layers
model=Init_Freeze_ShiftSubtract_Layers(model, wave, momentum_magnitude=momentum_magnitude)
# Now compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary
# model.summary()

from tensorflow.keras.utils import plot_model

# Assuming you've built your model as qvmodel
# plot_model(model, to_file="qvmodel.png", show_shapes=True, show_layer_names=True)
plot_model(model, to_file="qvmodel.png", show_shapes=True, show_layer_names=True)

"""**HYPERPARAMETERS AND DATA AUGMENTATION**"""

#data augmentation
datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

"""**MODEL FIT, LEARNING SCHEDULER AND MODEL TRAINING**"""

from tensorflow.keras.optimizers import Adam

learning_rate=0.01
batch_size=64
epochs=360

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 140, 200, 260 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """

    lr = 0.01
    # if epoch > 380:
    #     lr = 0.00001
    # if epoch > 380:
    #     lr = 0.00005
    if epoch > 260:
        lr = 0.0001
    elif epoch > 200:
        lr = 0.0005
    elif epoch > 140:
        lr = 0.001
    elif epoch > 80:
        lr = 0.005
    print('Learning rate: ', lr)
    return lr


from keras.callbacks import LearningRateScheduler


lr_scheduler = LearningRateScheduler(lr_schedule)


# Mount Google Drive
from google.colab import drive

# Save to local folder or your drive
# drive.mount('/content/drive')

from tensorflow.keras.callbacks import ModelCheckpoint

# Define the ModelCheckpoint callback
checkpoint_path = "/content/QVHeavyCNNtest_best.keras"  # Specify the path in your Google Drive

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)


# Train the model with both callbacks
callbacks_list = [checkpoint, lr_scheduler]


history=model.fit(datagen.flow(x_train, y_train,
              batch_size=batch_size),
              epochs=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)

"""**MODEL EVALUATE**"""

pretrained_model = tf.keras.models.load_model('/content/QVHeavyCNNtest_best.keras')

# Score trained model.
scores = pretrained_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
