import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import pandas as pd
import gdown  # for downloading the model
from .CNN import CNN  # make sure your CNN.py has a CNN class

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # flask_app folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

disease_info = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')

# -----------------------------
# Model setup
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt')
GDRIVE_FILE_ID = "1KQIgBiiX2e3OluN5B6sNFOSF0jdXVa9s"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)
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

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

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
    # For Render, do not set debug=True
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
