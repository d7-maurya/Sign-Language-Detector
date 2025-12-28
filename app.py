from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- LOAD MODEL & ENCODER ---
model = None
le = None
MODEL_LOADED = False

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    # Load the LabelEncoder (The translator that turns 0 -> 'A')
    le = model_dict.get('le', None)
    MODEL_LOADED = True
    print("✅ Model and LabelEncoder loaded successfully.")
except FileNotFoundError:
    print("⚠️ Model not found. Please run train_model.py first.")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'success': False, 'prediction': 'No Hand'})
        
        # Extract and Normalize Landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        
        for i in range(len(hand_landmarks.landmark)):
            # Flip X because webcam is mirrored
            data_aux.append(1.0 - hand_landmarks.landmark[i].x)
            data_aux.append(hand_landmarks.landmark[i].y)

        # Normalize relative to wrist (landmark 0)
        arr = np.array(data_aux).reshape(-1, 2)
        origin = arr[0]
        rel = arr - origin
        dists = np.linalg.norm(rel, axis=1)
        maxd = dists.max() if dists.max() != 0 else 1.0
        normalized = (rel / maxd).flatten()
        
        # Predict
        prediction = model.predict([np.asarray(normalized)])
        pred_raw = prediction[0]

        # --- CRITICAL FIX: CONVERT NUMBER TO LETTER ---
        if le:
            # Use the saved encoder to get the letter (e.g., 0 -> 'A')
            prediction_label = le.inverse_transform([int(pred_raw)])[0]
        else:
            # Fallback if no encoder found
            prediction_label = str(pred_raw)

        # Get Confidence
        try:
            proba = model.predict_proba([np.asarray(normalized)])
            confidence = float(proba.max()) * 100
        except:
            confidence = 100.0
        
        return jsonify({
            'success': True,
            'prediction': prediction_label,
            'confidence': confidence
        })
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)