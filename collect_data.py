import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 2. Setup Data Storage
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 3. User Input
class_name = input("Enter the name of the letter you want to collect (e.g., 'A'): ")
dataset_size = 100  # We will collect 100 samples per letter

cap = cv2.VideoCapture(0)

data = []  # To store the hand landmarks
labels = [] # To store the label (A, B, C...)

print(f"Collecting data for '{class_name}'. Press 'Q' when ready to start!")

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw the skeleton for visual feedback
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the instruction text
    cv2.putText(img, f"Collecting: {class_name}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, "Press 'Q' to Start", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    cv2.imshow("Data Collection", img)
    if cv2.waitKey(25) == ord('q'):
        break

# Start Collecting Loop
counter = 0
while counter < dataset_size:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract just the math coordinates (x, y)
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
            
            data.append(data_aux)
            labels.append(class_name)
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        counter += 1
        print(f"Sample: {counter}/{dataset_size}")

    cv2.putText(img, f"Saving... {counter}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow("Data Collection", img)
    cv2.waitKey(1)

# Save the data to a file
file_name = os.path.join(DATA_DIR, f"{class_name}.pickle")
f = open(file_name, 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"✅ Successfully saved {dataset_size} samples for '{class_name}'!")
cap.release()
cv2.destroyAllWindows()
