from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove
import numpy as np
import tensorflow as tf
tflite = tf.lite
import json
import io
import cv2

app = Flask(__name__)

# New CLAHE model (99.83%)
interpreter = tflite.Interpreter(model_path='models/plant_model_clahe.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open('models/class_labels_clahe.json') as f:
    labels = json.load(f)

print("✅ CLAHE model loaded!")

def apply_clahe(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_cv[:,:,0] = clahe.apply(img_cv[:,:,0])
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB))

def preprocess_input_mobilenet(img_array):
    img_array = img_array.astype(np.float32)
    img_array /= 127.5
    img_array -= 1.0
    return img_array

def process_image(file_bytes):
    # Remove background first
    try:
        removed = remove(file_bytes)
        img = Image.open(io.BytesIO(removed)).convert('RGBA')
        white_bg = Image.new('RGB', img.size, (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        img = white_bg
    except:
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')

    # Tight crop non-white pixels
    img_array = np.array(img)
    mask = ~((img_array[:,:,0] > 240) &
             (img_array[:,:,1] > 240) &
             (img_array[:,:,2] > 240))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]
        pad = 10
        h, w = img_array.shape[:2]
        rmin, rmax = max(0,rmin-pad), min(h,rmax+pad)
        cmin, cmax = max(0,cmin-pad), min(w,cmax+pad)
        img = img.crop((cmin, rmin, cmax, rmax))
    img = apply_clahe(img)
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = preprocess_input_mobilenet(np.array(img, dtype=np.float32))
    return np.expand_dims(img_array, axis=0)



@app.route('/')
def home():
    return "🌿 Floraverse CLAHE Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img_array = process_image(request.files['image'].read())
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    top3_idx = np.argsort(output[0])[::-1][:3]
    plant = labels[str(top3_idx[0])]
    confidence = round(float(output[0][top3_idx[0]]) * 100, 2)
    top3 = [{'plant': labels[str(i)], 'confidence': round(float(output[0][i])*100, 2)} for i in top3_idx]
    print(f"✅ Predicted: {plant} ({confidence}%)")
    return jsonify({'plant': plant, 'confidence': confidence, 'top3': top3})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)