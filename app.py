from flask import Flask, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.keras')

# Define the classes
classes = ['Bear', 'Newfoundland']

# Define the route for the homepage
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Newfoundland vs Bear: The Ultimate Classifier</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            .jumbotron {{
                background-color: rgba(233, 236, 239, 0.8);
                padding: 2rem 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="jumbotron text-center">
                <h1 class="display-4">Newfoundland vs Bear Classifier</h1>
                <p class="lead">Ever wondered if that furry friend is a giant dog or an actual bear? Don't worry, we got you covered! Simply upload a picture and we'll let you know whether to fetch 
the leash or run for your life!</p>
                <form action="/predict" method="POST" enctype="multipart/form-data" class="mt-4">
                    <div class="form-group">
                        <input type="file" name="file" class="btn btn-primary">
                    </div>
                    <button type="submit" class="btn btn-success">Upload and Predict</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

# Define the route for handling image uploads and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        class_name = classes[class_idx]
        
        os.remove(filepath)
        
        # Set the background image and text based on the prediction
        if class_name == 'Bear':
            background_image = url_for('static', filename='bearbg.jpeg')
            prediction_text = "It's a bear! Or maybe just a really fluffy dog? Better not take any chances!"
        else:
            background_image = url_for('static', filename='dogbg.jpeg')
            prediction_text = "Phew! It's just a Newfoundland. No need to panic, just give it a belly rub!"

        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Newfoundland vs Bear: The Ultimate Classifier</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{
                    background-image: url('{background_image}');
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                .jumbotron {{
                    background-color: rgba(233, 236, 239, 0.8);
                    padding: 2rem 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <div class="jumbotron text-center">
                    <h1 class="display-4">Newfoundland vs Bear Classifier</h1>
                    <p class="lead">{prediction_text}</p>
                    <form action="/predict" method="POST" enctype="multipart/form-data" class="mt-4">
                        <div class="form-group">
                            <input type="file" name="file" class="btn btn-primary">
                        </div>
                        <button type="submit" class="btn btn-success">Upload another image</button>
                    </form>
                </div>
            </div>
        </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

