import gradio as gr

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input

model = tf.keras.models.load_model('xray_model.h5')
class_names = ['NORMAL', 'PNEUMONIA']

sample_images = [
    ['IM-0105-0001.jpeg'],
    ['person104_bacteria_492.jpeg']
]

def predict_image(img):
    # Preprocessing the image
    x = tf.keras.preprocessing.image.img_to_array(img)
    X = np.array([x])  # Convert single image to a batch.
    X = preprocess_input(X)
    prediction=model.predict(X)[0].flatten()
    # Normalize the prediction
    prediction = (prediction - np.min(prediction))/np.ptp(prediction)
    return {class_names[i]: float(prediction[i]) for i in range(2)}

image = gr.inputs.Image(shape=(299,299))
label = gr.outputs.Label(num_top_classes=2)
iface = gr.Interface(fn=predict_image, 
             inputs=image, 
             outputs=label, 
             interpretation='default', 
             examples=sample_images,                     
             title = 'Pneumonia Classification on chest X-rays App',             
             description= 'Get classification for the input image among NORMAL and PNEUMONIA'
)
iface.launch()