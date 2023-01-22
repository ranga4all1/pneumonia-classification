#!/usr/bin/env python
# coding: utf-8

# Using the model
# 
# - Loading the model
# - Evaluating the model
# - Getting predictions

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input

# Variables
dataset_dir = './data/chest_xray/'
test_dir = os.path.join(dataset_dir, 'test')
class_names = ['NORMAL', 'PNEUMONIA']

# data generators

print("Generating test data....")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

print("Data Generated !!")

# load model
print("Loading Model....")
model = keras.models.load_model('xray_model.h5')

# Evaluate
print("Testing scores is...")
model.evaluate(test_ds)

# Predictions
print(" predicting image 1... ")

path = './data/chest_xray/test/NORMAL/IM-0105-0001.jpeg'
img = load_img(path, target_size=(299, 299))
x = tf.keras.preprocessing.image.img_to_array(img)
X = np.array([x])  # Convert single image to a batch.
X = preprocess_input(X)
prediction=model.predict(X)[0].flatten()
prediction = (prediction - np.min(prediction))/np.ptp(prediction)
print({class_names[i]: float(prediction[i]) for i in range(2)})

print("Prediction completed!")


print(" predicting image 2... ")

path = './data/chest_xray/test/PNEUMONIA/person104_bacteria_492.jpeg'
img = load_img(path, target_size=(299, 299))
x = tf.keras.preprocessing.image.img_to_array(img)
X = np.array([x])  # Convert single image to a batch.
X.shape
X = preprocess_input(X)
prediction=model.predict(X)[0].flatten()
prediction = (prediction - np.min(prediction))/np.ptp(prediction)
print({class_names[i]: float(prediction[i]) for i in range(2)})

print("Prediction completed!")