from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_cors import CORS
#from PIL import Image
#print("Done with flask imports")
from function import *
from function import get_selftrained, get_finetunned, get_signlanguage, preprocess_image
#print("impoted funtions")

app = Flask(__name__)
CORS(app)

print(" * Loading Keras models...")
selftrained_model = get_selftrained()
finetunned_model = get_finetunned()
signlanguage_model = get_signlanguage()
print(" * Models Loaded...")


@app.route('/')
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/selftrained', methods=['GET'])
def selftrained():
    return render_template('selftrained.html')

@app.route('/finetunned', methods=['GET'])
def finetunned():
    return render_template('finetunned.html')

@app.route('/signlanguage', methods=['GET'])
def signlanguage():
    return "<b>THIS IS SIGN LANGUAGE</b>"

@app.route("/predict_selftrained", methods=["POST"])
def predict_selftrained():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(244, 244))
    #print('image pre processed')
    
    prediction = selftrained_model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    print("Response: ", response)
    #print("Respponse returned")
    return jsonify(response)

@app.route("/predict_finetunned", methods=["POST"])
def predict_finetunned():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    #print('image pre processed')
    
    prediction = finetunned_model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    print("Response: ", response)
    #print("Respponse returned")
    return jsonify(response)

@app.route("/predict_signlanguage", methods=["POST"])
def predict_signlanguage():
    pass

#if __name__ == '__main__':
#    app.run()
