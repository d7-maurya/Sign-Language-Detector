import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("❌ Error: 'model.p' not found. Did you run train_model.py?")
    exit()

# 2. Setup MediaPipe
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# max_num_hands=1 ensures we don't get the "84 features" error
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

print("✅ System Started. Press 'q' to exit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # --- THE FIX IS HERE ---
        # We are now collecting RAW coordinates exactly like collect_data.py did
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            
            x_.append(x)
            y_.append(y)

            # Previously, we subtracted min(x_) here. REMOVED that.
            data_aux.append(x) 
            data_aux.append(y)

        # Safety Check
        if len(data_aux) == 42:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                # Draw the Box and Text
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            except Exception as e:
                pass # Ignore errors for smooth video

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
