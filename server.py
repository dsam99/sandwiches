from flask import (
    Flask,
    jsonify,
    render_template
)

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

@app.route('/summary')
def summary():
    d = {
        'hey': 3
    }
    return jsonify(d)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)