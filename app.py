from flask import Flask, request, jsonify
from flask_cors import CORS
from src.inference import load_model, predict
from PIL import Image
import io, os

app = Flask(__name__)
CORS(app)

model = load_model()

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'status':'error','message':'No image provided'}), 400
    file = request.files['image'].read()
    img = Image.open(io.BytesIO(file)).convert("RGB")
    tmp_path = "tmp.jpg"
    img.save(tmp_path)
    try:
        res = predict(tmp_path, model)
        os.remove(tmp_path)
        return jsonify({'status':'success','result': res})
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 500


@app.route('/')
def home():
    return "Welcome to Deep Fake Image Detector",200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
