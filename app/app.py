from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model.h5"
IMAGE_SIZE = (224, 224)
generator = {'Grape__black_measles': 0, 'Grape__black_rot': 1, 'Grape__healthy': 2, 'Grape__leaf_blight_(isariopsis_leaf_spot)': 3, 'Lemon__diseased': 4, 'Lemon__healthy': 5, 'Pomegranate__diseased': 6, 'Pomegranate__healthy': 7, 'guava_Disease Free': 8, 'guava_Phytopthora': 9, 'guava_Red rust': 10, 'guava_Scab': 11, 'guava_Styler and Root': 12, 'mango_Anthracnose': 13, 'mango_Bacterial Canker': 14, 'mango_Cutting Weevil': 15, 'mango_Die Back': 16, 'mango_Gall Midge': 17, 'mango_Healthy': 18, 'mango_Powdery Mildew': 19, 'mango_Sooty Mould': 20}
LABEL_MAP = {v: k for k, v in generator.items()}  
NUM_CLASSES = len(LABEL_MAP)

model = load_model(MODEL_PATH)
print("Model loaded successfully.")

def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/submit', methods=['POST'])
def submit():
    if 'file' not in request.files:
        return 'No file provided'
    
    file = request.files['file']
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = LABEL_MAP[predicted_class_idx]
        
        return render_template('disease.html', Disease = predicted_class)
    
    except Exception as e:
        return "Error while predicting"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = LABEL_MAP[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': {LABEL_MAP[i]: float(predictions[0][i]) for i in range(NUM_CLASSES)}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'})

@app.route("/")
def welcome():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)