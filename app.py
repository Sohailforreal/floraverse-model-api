from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import json
import tflite_runtime.interpreter as tflite

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
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = str(np.argmax(output))
    confidence = float(np.max(output))
    plant_name = class_labels[predicted_index]
    return jsonify({'plant': plant_name, 'confidence': round(confidence*100, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
