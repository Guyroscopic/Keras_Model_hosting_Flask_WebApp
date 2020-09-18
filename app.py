from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_cors import CORS

from forms import PhotoInputForm

#from function import *
from function import get_selftrained, get_finetunned, get_signlanguage, preprocess_image, Image, base64, io

import os
from OpenSSL import SSL

#---------------------------------------------   APP CONFIG    ---------------------------------------------#

app = Flask(__name__)
GET, POST = "GET", "POST"
CORS(app)
app.config['SECRET_KEY'] = 'I have a dream'

#---------------------------------------------  LOADING MODELS  --------------------------------------------#

print(" * Loading Keras models...")
selftrained_model = get_selftrained()
finetunned_model = get_finetunned()
signlanguage_model = get_signlanguage()
print(" * Models Loaded...")

#---------------------------------------------   ROUTES   --------------------------------------------------#
@app.route('/')
@app.route('/home', methods=[GET])
def home():
    return render_template('home.html')

@app.route('/selftrained', methods=[GET, POST])
def selftrained():
    
    form = PhotoInputForm()    
    if form.validate_on_submit():
        photo = form.photo.data

        image = Image.open(photo)
        processed_image = preprocess_image(image, target_size=(244, 244))
        prediction = selftrained_model.predict(processed_image).tolist()

        response = {'dog': format(prediction[0][0]*100, '.2f')+'%', 'cat': format(prediction[0][1]*100, '.2f')+'%'}        
        
        return render_template('selftrained.html', form=form, response=response)
    
    return render_template('selftrained.html', form=form)

@app.route('/finetunned', methods=[GET])
def finetunned():
    return render_template('finetunned.html')

@app.route('/signlanguage', methods=[GET])
def signlanguage():
    return "<b>THIS IS SIGN LANGUAGE</b>"

@app.route('/mnistclassifier', methods=[GET])
def mnistclassifier():
    return render_template('MNIST Classifier.html')

@app.route('/tfjsrps' , methods=[GET])
def tfjsrps():
    return render_template('retrain.html')

@app.route("/predict_selftrained", methods=[POST])
def predict_selftrained():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(244, 244))
    
    prediction = selftrained_model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    print("Response: ", response)
    return jsonify(response)

@app.route("/predict_finetunned", methods=[POST])
def predict_finetunned():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    prediction = finetunned_model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': format(prediction[0][0]*100, '.2f')+'%',
            'cat': format(prediction[0][1]*100, '.2f')+'%'
        }
    }
    print("Response: ", response)
    return jsonify(response)

@app.route("/predict_signlanguage", methods=[POST])
def predict_signlanguage():
    pass

#--------------------------------------RUNNING APP----------------------------#
if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=6969,
            ssl_context=('cert.pem', 'key.pem')
    )
