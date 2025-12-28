import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = './data'

data = []
labels = []


def normalize_sample(sample):
    # sample: flat list [x0,y0,x1,y1,...]
    arr = np.array(sample).reshape(-1, 2)
    # use wrist (landmark 0) as origin
    origin = arr[0]
    rel = arr - origin
    # scale by max euclidean distance to keep values in similar range
    dists = np.linalg.norm(rel, axis=1)
    maxd = dists.max()
    if maxd == 0:
        maxd = 1.0
    norm = (rel / maxd).flatten()
    return norm.tolist()


# 1. Load all the data files
print("Loading data and normalizing...")
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pickle"):
        file_path = os.path.join(DATA_DIR, filename)

        with open(file_path, 'rb') as f:
            dict_data = pickle.load(f)

        # Normalize each sample to match runtime preprocessing
        for s in dict_data['data']:
            data.append(normalize_sample(s))
        labels.extend(dict_data['labels'])

# Convert to numpy arrays for the AI to understand
data = np.array(data)
labels = np.array(labels)

# Encode string labels to integers
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# 2. Split into Training and Test sets
# We use 80% of data to teach, and 20% to test if it learned correctly
x_train, x_test, y_train, y_test = train_test_split(data, labels_enc, test_size=0.2, shuffle=True, stratify=labels_enc)

# 3. Train the Model
print("Training the model... (This might take a few seconds)")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

# 4. Test Accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"Model Trained! Accuracy: {score * 100:.2f}%")

# 5. Save the trained model and label encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'le': le}, f)

print("Model and label encoder saved as 'model.p'")
