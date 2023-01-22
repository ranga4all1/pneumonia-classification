#!/usr/bin/env python
# coding: utf-8

# Pneumonia Classification on chest X-rays -- Model Training
import os
import mlnotify

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report,confusion_matrix

print("Libraries Imported !!")

print('tf_version:', tf.__version__)
print('device_type:', tf.config.list_physical_devices('GPU'))

# set seed
tf.random.set_seed(1)

# Variables
dataset_dir = './data/chest_xray/'

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
class_names = os.listdir(train_dir)

COUNT_NORMAL = 1349
COUNT_PNEUMONIA = 3884

initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

TRAIN_IMG_COUNT = COUNT_NORMAL + COUNT_PNEUMONIA
weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))


# Parameters
input_size = 299
batch_size = 32
learning_rate = 0.0005
size_inner = 128
droprate = 0.2
n_epochs = 20

# data generatores

print("Generating data....")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2     # set validation split
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Flow training images in batches using train_datagen generator
train_ds = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(input_size, input_size),  # All images will be resized
        batch_size=batch_size,
        subset='training'      # set as training data
)

# Flow val images in batches using val_datagen generator
val_ds = train_datagen.flow_from_directory(
        train_dir,  # same directory as training data
        target_size=(input_size, input_size),  # All images will be resized
        batch_size=batch_size,
        subset='validation'     # set as validation data
)

# Flow test images in batches using test_datagen generator
test_ds = test_datagen.flow_from_directory(
        test_dir,  # This is the source directory for test images
        target_size=(input_size, input_size),  # All images will be resized
        batch_size=batch_size,
        shuffle=False
)

print("Data Generated !!")

# Getting class names
classes = train_ds.class_indices
classes = list(classes.keys())
print("The classes are : ",classes)



# Build model
print("Building Model....")

def make_model(input_size=299, learning_rate=0.001, size_inner=512,
               droprate=0.2):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(2)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
 
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# finetuning steps

# Defining callbacks

# Checkpoint callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "xray_model.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Defining early stopping to prevent overfitting
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    min_delta = 0,
    patience = 2,
    verbose = 2,
    restore_best_weights = True
)

# Exponential learning rate decay
initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

# Build model 

model = make_model(
    input_size=input_size,
    learning_rate=lr_schedule,
    size_inner=size_inner,
    droprate=droprate
)

print("Model build completed!")

# Model Training
print("Starting Training . . .")

history = model.fit(
    train_ds,
    # steps_per_epoch=130,  # Total train images = batch_size * steps
    epochs=n_epochs,
    validation_data=val_ds,
    # validation_steps=32,  # Total val images = batch_size * steps
    class_weight=class_weight,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

print("Finished Training!")

# Test model
print("Testing scores is...")
model.evaluate(test_ds, verbose=2, return_dict=True)

# Making predictions
print("Making predictions . . .")
preds = model.predict(test_ds, verbose=2)
predicted_class_indices=np.argmax(preds,axis=1)
print("Done Predicting!!")

# Actual values
actual = test_ds.classes

# classification report
print("Classification Report ::\n")
print(classification_report(actual, predicted_class_indices))
print("\n\n")

# Confusion matrix
print("Confusion Matrix ::\n")
print(confusion_matrix(actual, predicted_class_indices))
