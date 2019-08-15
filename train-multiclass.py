"""
First, you need to collect training data and deploy it like this.
e.g. 3-classes classification Pizza, Poodle, Rose

  ./data/
    train/
      pizza/
        pizza1.jpg
        pizza2.jpg
        ...
      poodle/
        poodle1.jpg
        poodle2.jpg
        ...
      rose/
        rose1.jpg
        rose2.jpg
        ...
    validation/
      pizza/
        pizza1.jpg
        pizza2.jpg
        ...
      poodle/
        poodle1.jpg
        poodle2.jpg
        ...
      rose/
        rose1.jpg
        rose2.jpg
        ...
"""

import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 1

train_data_path = r'.\training'
validation_data_path = r'.\validation'


"""
Parameters
"""
img_width, img_height = 600,600
batch_size = 32
samples_per_epoch = 1000
validation_steps = 400
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 3
lr = 0.0004

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rotation_range=10,shear_range=0.2,
        zoom_range=0.2,brightness_range=(0,3))

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


filepath=r'.\ckpt\weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
                             mode='auto', save_weights_only=True, period=10)
model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    callbacks=[checkpoint])

model.save_weights(r'.\models\weights.h5')
