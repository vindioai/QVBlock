
"""**QV VIT**"""

# Quantum Vision Theory in Deep Learning for Object Recognition
# Copyright (C) 2025 [Vindio AI Software Ltd. - https://vindioai.com / Cem Direkoglu and Melike Sah]
# AGPL-3.0 Licence

# QV-ViT Model - QV Block is Integrated to the ViT Model (QV-ViT-8/8 Model in Our Paper)


# Licensing Summary

# | Use Case                                   | Free? | Must Release Source Code?      | License Type           |
# |--------------------------------------------|-------|--------------------------------|------------------------|
# | Academic / research (non-commercial)       | Yes   |  Yes (if modified/distributed) | AGPL-3.0               |
# | Personal experiments / hobby projects      | Yes   |  Yes (if shared publicly)      | AGPL-3.0               |
# | Open-source commercial product             | Yes   |  Yes                           | AGPL-3.0               |
# | Closed-source commercial product           | No    |  No (with Enterprise License)  | Enterprise License     |

# If you use this work, please cite our paper:

# C. Direkoglu and M. Sah, "Quantum Vision Theory in Deep Learning for Object Recognition," IEEE Access, vol. 13, pp. 132194-132208, 2025. doi: 10.1109/ACCESS.2025.3592037


# First pip install vindioai
import vindioai
from vindioai import QVBlock, Init_Freeze_ShiftSubtract_Layers

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow
import numpy as np
import zipfile

from tensorflow.keras import initializers
from tensorflow.keras.datasets import cifar10
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import tensorflow.keras.activations
from tensorflow.keras.layers import BatchNormalization


import tensorflow as tf

import tensorflow
import numpy as np
import zipfile


tf.keras.backend.clear_session()

"""**LOADING DATASET AND SETTING VIT HYPERPARAMETERS**"""

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


# //////ViT hyperparameters

image_size = 64  # We'll resize input images to this size
patch_size = 8  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64

num_heads = 4
# num_heads = 12

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

transformer_layers = 8
# transformer_layers = 12


mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image)
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy())
    plt.axis("off")


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

"""**Integrate QVBLOCK to a ViT Model**"""

# input = Input(shape=(64,64,3), name='input')

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

    # Create patches.
patches = Patches(patch_size)(infowave)
    # Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
        # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=projection_dim, dropout=0.1
      )(x1, x1)
        # Skip connection 1.
      x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
      x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
      encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dropout(0.5)(representation)
    # Add MLP.
features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
logits = layers.Dense(num_classes, activation = 'softmax')(features)

# Create the Keras model.

model = Model(inputs=inputs, outputs=logits)

# Must initialize and freeze the weight learning for shift and subtract conv layers
model=Init_Freeze_ShiftSubtract_Layers(model, wave, momentum_magnitude=momentum_magnitude)

print(model.summary())


from tensorflow.keras.utils import plot_model

# Assuming you've built your model as qvmodel
# plot_model(model, to_file="qvmodel.png", show_shapes=True, show_layer_names=True)
plot_model(model, to_file="qvmodel.png", show_shapes=True, show_layer_names=True)

# ###############################

tf.keras.utils.plot_model(
  model,
  #    to_file='model.png',
  show_shapes=False,
  show_dtype=False,
  show_layer_names=True,
  rankdir='TB',
  expand_nested=False,
  dpi=96,
  layer_range=None,
  show_layer_activations=False
  )

"""**HYPERPARAMETERS AND DATA AUGMENTATION**"""

learning_rate = 0.001
batch_size = 64
epochs = 360


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


# print('AFTER DATA AUGMENTATION x_train shape:', x_train.shape)

"""**MODEL FIT, LEARNING SCHEDULER AND MODEL TRAINING**"""

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Nadam


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 140, 200, 260 and 320 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """


    lr = 0.001
    if epoch > 320:
        lr = 0.000005
    elif epoch > 260:
        lr = 0.00001
    elif epoch > 200:
        lr = 0.00005
    elif epoch > 140:
        lr = 0.0001
    elif epoch > 80:
        lr = 0.0005
    print('Learning rate: ', lr)
    return lr



from keras.callbacks import LearningRateScheduler


lr_scheduler = LearningRateScheduler(lr_schedule)


from google.colab import drive

# Mount Drive or save local folder
# drive.mount('/content/drive')


from tensorflow.keras.callbacks import ModelCheckpoint

# Define the ModelCheckpoint callback
checkpoint_path = "/content/QV_ViT_best.keras"  # Specify the path in your Google Drive

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)


# Train the model with callbacks
callbacks_list = [checkpoint, lr_scheduler]

history=model.fit(datagen.flow(x_train, y_train,
              batch_size=batch_size),
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)

"""# Properly Load the Pre-Trained QV-ViT Model After Training"""

# //////ViT hyperparameters

image_size = 64  # We'll resize input images to this size
patch_size = 8  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64

num_heads = 4
# num_heads = 12

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

transformer_layers = 8
# transformer_layers = 12


mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image)
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy())
    plt.axis("off")


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Then use the exactly same QV-ViT Model Architecture that you trained your model on

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

    # Create patches.
patches = Patches(patch_size)(infowave)
    # Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
        # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=projection_dim, dropout=0.1
      )(x1, x1)
        # Skip connection 1.
      x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
      x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
      encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dropout(0.5)(representation)
    # Add MLP.
features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
logits = layers.Dense(num_classes, activation = 'softmax')(features)

# Create the Keras model.

pretrainedmodel = Model(inputs=inputs, outputs=logits)

# Load the pre-trained model weights
pretrainedmodel.load_weights('/content/QV_ViT_best.keras')

learning_rate = 0.001
batch_size = 64
epochs = 360

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Nadam


# Do not forget to compile the pre-trained model before you evaluate with test samples
pretrainedmodel.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


"""# Evaluate Using the Pre-Trained QV-ViT Model"""

scores = pretrainedmodel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])