import numpy as  np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import io, base64

def get_selftrained():
    selftrained_model = load_model('TestCNN1.h5')
    return selftrained_model

def get_finetunned():
    finetunned_model = load_model('finetunned_trained.h5')
    return finetunned_model

def get_signlanguage():
    return 0

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image
