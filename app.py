from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import json
import tflite_runtime.interpreter as tflite
from rembg import remove
from PIL import Image
import io


app = Flask(__name__)

interpreter = tflite.Interpreter(model_path='models/plant_model.tflite')
interpreter.allocate_tensors()

with open('models/class_labels.json') as f:
    class_labels = json.load(f)

@app.route('/')
def home():
    return "🌿 Floraverse Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Open image
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Center crop to square
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Remove background
    result = remove(img)
    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)
    white_bg = Image.new('RGB', result.size, (255, 255, 255))
    white_bg.paste(result, mask=result.split()[3])
    img = white_bg
    
    # Resize and predict
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_index = str(np.argmax(output))
    confidence = float(np.max(output)) * 100
    plant_name = class_labels[predicted_index]
    
    return jsonify({'plant': plant_name, 'confidence': round(confidence, 2)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
