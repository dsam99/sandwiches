from flask import (
    Flask,
    jsonify,
    render_template
)
from flask import request
from base64_to_img import convert_and_save
import base64

# Create the application instance
app = Flask(__name__, template_folder="templates")

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
    convert_and_save(base64.b64decode(b64_string), "jpeg")
    # process it
    # delete the image
    d = {
        'success': True,
        'type': 1,
    }
    return jsonify(d)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)