from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import json
import io

app = Flask(__name__)

# Final 30-plant model (98.83%)
interpreter30 = tflite.Interpreter(model_path='models/plant_model_final.tflite')
interpreter30.allocate_tensors()
input30 = interpreter30.get_input_details()
output30 = interpreter30.get_output_details()
with open('models/class_labels_final.json') as f:
    labels30 = json.load(f)

# 7-plant model (97.35%)
interpreter7 = tflite.Interpreter(model_path='models/plant_model_balanced.tflite')
interpreter7.allocate_tensors()
input7 = interpreter7.get_input_details()
output7 = interpreter7.get_output_details()
with open('models/class_labels_balanced.json') as f:
    labels7 = json.load(f)

print("✅ Both models loaded!")

def process_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w-min_dim)//2, (h-min_dim)//2,
                    (w+min_dim)//2, (h+min_dim)//2))
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

@app.route('/')
def home():
    return "🌿 Floraverse Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img_array = process_image(request.files['image'].read())
    interpreter30.set_tensor(input30[0]['index'], img_array)
    interpreter30.invoke()
    output = interpreter30.get_tensor(output30[0]['index'])
    plant = labels30[str(np.argmax(output))]
    confidence = round(float(np.max(output)) * 100, 2)
    print(f"✅ 30-plant: {plant} ({confidence}%)")
    return jsonify({'plant': plant, 'confidence': confidence})

@app.route('/predict7', methods=['POST'])
def predict7():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img_array = process_image(request.files['image'].read())
    interpreter7.set_tensor(input7[0]['index'], img_array)
    interpreter7.invoke()
    output = interpreter7.get_tensor(output7[0]['index'])
    plant = labels7[str(np.argmax(output))]
    confidence = round(float(np.max(output)) * 100, 2)
    print(f"✅ 7-plant: {plant} ({confidence}%)")
    return jsonify({'plant': plant, 'confidence': confidence})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)