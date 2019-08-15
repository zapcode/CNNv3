import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D


img_width, img_height = 400,400
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
model_weights_path = r'.\models\weights.h5'

filename=r'C:\Users\acer\Desktop\CNN-Image-Classifier-master\src\training\washington\washington.jpg'


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  print(array)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Cal")
  elif answer == 1:
    print("Labels: Illi")
  elif answer == 2:
    print("Label: Washington")

  return answer


def model():
  model = Sequential()
  model.add(
    Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode="same", input_shape=(img_width, img_height, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

  model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(classes_num, activation='softmax'))
  return model

model = model()
model.load_weights(model_weights_path)
predict(filename)

