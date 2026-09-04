import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

FERPLUS_CLASS_MAPPING = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Surprise',
    3: 'Sadness',
    4: 'Anger',
    5: 'Disgust',
    6: 'Fear',
    7: 'Contempt',
}

EMOTION_COLORS = {
    'Neutral':   '#6c757d',
    'Happiness': '#ffc107',
    'Surprise':  '#17a2b8',
    'Sadness':   '#007bff',
    'Anger':     '#dc3545',
    'Disgust':   '#6f42c1',
    'Fear':      '#fd7e14',
    'Contempt':  '#20c997',
}

# Load TFLite model once at startup
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model/ferplus_model_pd_best.tflite')
    interpreter.allocate_tensors()
    return interpreter

model = load_model()
input_details  = model.get_input_details()
output_details = model.get_output_details()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face_img):
    """Resize and normalize a grayscale face crop to (1, 48, 48, 1)."""
    face = cv2.resize(face_img, (48, 48)).astype(np.float32) / 255.0
    return face.reshape(1, 48, 48, 1)


def tflite_predict(face_input):
    """Run a single (1,48,48,1) float32 array through the TFLite model."""
    model.set_tensor(input_details[0]['index'], face_input)
    model.invoke()
    return model.get_tensor(output_details[0]['index'])[0]


def decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL into a BGR numpy array."""
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_inference(bgr_img: np.ndarray):
    """Detect faces, classify emotions, return list of result dicts."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    annotated = bgr_img.copy()

    for (x, y, w, h) in faces:
        face_crop = gray[y:y + h, x:x + w]
        inp   = preprocess_face(face_crop)
        probs = tflite_predict(inp)
        top_idx = int(np.argmax(probs))
        emotion = FERPLUS_CLASS_MAPPING[top_idx]
        confidence = float(probs[top_idx])

        all_scores = {FERPLUS_CLASS_MAPPING[i]: float(probs[i]) for i in range(len(probs))}

        color_hex = EMOTION_COLORS.get(emotion, '#ffffff')
        color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2)
        label = f'{emotion} {confidence:.0%}'
        cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        results.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'emotion': emotion,
            'confidence': round(confidence * 100, 1),
            'scores': {k: round(v * 100, 1) for k, v in all_scores.items()},
            'color': color_hex,
        })

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    return results, annotated_b64


@app.route('/')
def index():
    return render_template('index.html', emotions=EMOTION_COLORS)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img = decode_image(data['image'])
    except Exception:
        return jsonify({'error': 'Invalid image data'}), 400

    results, annotated = run_inference(img)
    return jsonify({'faces': results, 'annotated': annotated})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
