from urllib import request

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

from settings import token

BASE_FILES_URL = f'https://api.telegram.org/file/bot{token}'
IMAGE_SIZE = (32, 32)

model = load_model('data/svhn_model.h5')


def download_image(file_path):
    url = f'{BASE_FILES_URL}/{file_path}'
    with request.urlopen(url) as res:
        image = np.asarray(bytearray(res.read()), dtype='uint8')
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        image = cv.resize(image, IMAGE_SIZE)
        return image


def predict(image):
    X = np.asarray(image).astype('float32')
    X = X.reshape(1, 32, 32, 3)
    predictions = model.predict_classes(X)
    return (predictions[0] + 1) % 10
