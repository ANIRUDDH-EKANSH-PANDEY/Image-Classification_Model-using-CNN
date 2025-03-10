import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load your trained model
model = load_model("./static/cifar10_enhanced_cnn_model_1.h5",compile=False)
categories = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def preprocess_image(image_path):
    """Preprocess the uploaded image."""
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/")
def index():
    """Home page for image upload."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]
    if file.filename == "":
        return "No selected file."

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = round(predictions[0][class_index] * 100, 2)

    result = {"class": categories[class_index], "confidence": confidence}
    return render_template("result.html", result=result, file_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
