from flask import (
    Flask,
    jsonify,
    render_template
)
from flask import request
from base64_to_img import convert_and_save
from model import (
    predict_class,
    load_model
)
import base64

import random

# Create the application instance
app = Flask(__name__, template_folder="templates")

MODEL_FILENAME = "sandwich_model_dropout2.h5"
#model = load_model(MODEL_FILENAME)

# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify():
    json = request.get_json(force=True)
    b64_string = json['image']

    # save the image
    #print(base64.b64encode(b64_string))
    # print(base64.b64decode())
    convert_and_save(b64_string, "jpg")#base64.b64decode(b64_string), "jpg")
    # process it
    img_filename = "tmp/imageToSave.jpg"
    #cube_type = predict_class(model, img_filename)
    cube_type = random.randint(0, 6)
    # delete the image
    d = {
        'success': True,
        'type': cube_type
    }
    return jsonify(d)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)
