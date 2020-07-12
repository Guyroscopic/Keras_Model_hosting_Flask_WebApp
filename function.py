#print("importing functions")

import numpy as  np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

def get_selftrained():
    #global model
    selftrained_model = load_model('TestCNN1.h5')
    return selftrained_model

def get_finetunned():
    #global model
    finetunned_model = load_model('finetunned_trained.h5')
    return finetunned_model

def get_signlanguage():
    pass

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    #print("RESIZING IMGAE")
    image = np.expand_dims(image, axis=0)

    return image
