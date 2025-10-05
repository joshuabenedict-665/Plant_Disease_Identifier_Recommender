import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import pandas as pd
from .CNN import CNN  # Ensure your CNN.py has CNN class accepting K parameter

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # flask_app folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.csv')
SUPPLEMENT_INFO_PATH = os.path.join(BASE_DIR, 'supplement_info.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt')

# -----------------------------
# Load data
# -----------------------------
disease_info = pd.read_csv(DISEASE_INFO_PATH, encoding='cp1252')
supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH, encoding='cp1252')

# -----------------------------
# Model setup
# -----------------------------
NUM_CLASSES = len(disease_info)
model = CNN(K=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    with torch.no_grad():
        output = model(input_data)
    output = output.numpy()
    return np.argmax(output)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        pred = prediction(file_path)

        # Disease info
        title = disease_info.at[pred, 'disease_name']
        description = disease_info.at[pred, 'description']
        prevent = disease_info.at[pred, 'Possible Steps']
        image_url = disease_info.at[pred, 'image_url']

        # Supplement info
        supplement_name = supplement_info.at[pred, 'supplement name']
        supplement_image_url = supplement_info.at[pred, 'supplement image']
        supplement_buy_link = supplement_info.at[pred, 'buy link']

        return render_template('submit.html',
                               title=title,
                               desc=description,
                               prevent=prevent,
                               image_url=image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image_url,
                               buy_link=supplement_buy_link)
    return redirect('/')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
