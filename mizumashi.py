import keras
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import sys
import random
from setuptools.sandbox import save_path


DATA_DIR = '/Users/tamotomina/Desktop/Owlimages'
CATEGORIES = ["キンメフクロウ", "アオバズク", "コミミズク", "コノハズク", "メンフクロウ","オオコノハズク", "シマフクロウ", "トラフズク", "ウラルフクロウ", "ワシミミズク", "シロフクロウ"]

SAVE_DIR = os.path.join(
   '/Users/tamotomina/Desktop/Owlimages',"owlimages2")
IMG_SIZE = 50
training_data = []

datagen = ImageDataGenerator(
   rotation_range=90,
   width_shift_range=0.3,
   height_shift_range=0.3,
   channel_shift_range=40.0,
   shear_range=0.39,
   zoom_range=[0.7, 1.3],
   horizontal_flip=True,
   vertical_flip=True
)

copy_num = 50


def create_training_data():

for class_num, category in enumerate(CATEGORIES):
  path = os.path.join(DATA_DIR, category)

  save_path = os.path.join(SAVE_DIR, category)

  if not os.path.exists(save_path):
   os.makedirs(save_path)

  for image_name in os.listdir(path):
    try:
      img = cv2.imread(os.path.join(path, image_name))
      img_resize = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      img_array = image.img_to_array(img_resize)
      img_array = np.expand_dims(img_array, axis=0)

      j = 0
      for d in datagen.flow(img_array, batch_size=1,
                            save_to_dir=save_path,
                            save_prefix=category,
                            save_format='jpeg'
                            ):
        j += 1
        d = np.resize(d, (d.shape[1], d.shape[2], d.shape[3]))
        training_data.append([d, class_num])
        if j == copy_num:
          break
    except Exception as e:
      pass


create_training_data()

random.shuffle(training_data)

X = []
y = []

for feature, label in training_data:
X.append(feature)
y.append(label)

X = np.array(X)
y = np.array(y)

np.savez('/Users/tamotomina/Desktop/Owlimages', X, y)　