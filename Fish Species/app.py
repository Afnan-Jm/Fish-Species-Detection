from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'  # Ensure 'static/upload' folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('FishModelClassifier_VGG161.h5', compile=False)

# Define class names
class_name = [
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 
    'Fourfinger Threadfin','Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 
    'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon', 
    'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish','Mosquito Fish', 
    'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
]

# Define fish information (example)
fish_info = {
    'Bangus': 'Bangus, also known as milkfish, is the national fish of the Philippines.',
    'Big Head Carp': 'Big Head Carp is a species of freshwater fish native to East Asia.',
    'Black Spotted Barb': 'Black Spotted Barb is a small freshwater fish found in Southeast Asia.',
    # Add more information as needed for each class
}

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    prediction = model.predict(img)
    y_class = prediction.argmax(axis=-1)
    res = class_name[int(y_class)]
    return res

@app.route('/', methods=['GET', 'POST'])
def load_file():
    return render_template('index.html')




@app.route('/contact', methods=['GET', 'POST'])
def upload_e():
    return render_template('contact.html')

# @app.route('/blog', methods=['GET', 'POST'])
# def uplo_e():
#     return render_template('blog.html')  
@app.route('/gallery', methods=['GET', 'POST'])
def upad_file():
    return render_template('gallery.html')


@app.route('/about', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict(filepath)
            fish_information = fish_info.get(prediction, 'Information not available')
            return render_template('blog.html', image_url=filename, prediction=prediction, fish_info=fish_information)
    return render_template('about.html')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
