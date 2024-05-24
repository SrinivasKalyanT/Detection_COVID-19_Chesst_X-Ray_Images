from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model (adjust the path as necessary)
model = tf.keras.models.load_model('model_detect.h5')

# Define class labels
class_labels = ['COVID', 'NORMAL','PNEUMONIA']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction_text='No file part')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file')

    if file:
        try:
            # Read the image and convert to RGB
            img = Image.open(file).convert('RGB')
            img = img.resize((128, 128))  # Resize to match the input shape of your model
            img_array = image.img_to_array(img)  # Convert image to numpy array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize pixel values

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_label = class_labels[predicted_class_index]

            return render_template('index.html', prediction_text=f'Prediction: {predicted_class_label}')
        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('index.html', prediction_text='Error processing image')

if __name__ == '__main__':
    app.run(debug=True)
