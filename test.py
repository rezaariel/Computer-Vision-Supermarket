import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = 'model_supermarket.h5'
model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_TO_TEST = 'dataset\care\IMG_20220318_192542.jpg'
class_names = sorted(os.listdir(RAW_DIR))

if os.path.exists(IMAGE_TO_TEST):

    img = image.load_img(IMAGE_TO_TEST, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    highest_score = np.argmax(prediction[0])
    prediction_name = class_names[highest_score]

    print(f'\nPrediction Result: This is "{prediction_name}"')

else:

    print(f'\nFile Not found "{IMAGE_TO_TEST}" Tidak ditemukan ')

