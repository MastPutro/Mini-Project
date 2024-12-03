from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

app = Flask(__name__)

# Load pretrained model and features
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = np.load("features.npy")
image_paths = np.load("image_paths.npy")

# Directory for uploaded images
UPLOAD_FOLDER = './static/uploaded/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return model.predict(img_array).flatten()

def find_similar_images(query_image_path, model, features, image_paths, top_n=10):
    query_features = extract_features(query_image_path, model)
    similarities = cosine_similarity([query_features], features)[0]
    indices = np.argsort(similarities)[::-1][:top_n]
    return [(image_paths[i], similarities[i]) for i in indices]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Find similar images
        similar_images = find_similar_images(filepath, model, features, image_paths)

        # Clean up folder for next use
        if len(os.listdir(UPLOAD_FOLDER)) > 10:
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)

        return render_template("index.html", query_image=file.filename, results=similar_images)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
