# backend/predictor.py

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ml_model import MLP
from clip_utils import image_to_clip_embedding

# Modellparameter
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.5
INPUT_DIM = 512

# Laden des Modells und der Preprozessoren
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("best_model.pt", map_location=device)
preproc = torch.load("preprocessors.pt", map_location=device, weights_only=False)

encoder: LabelEncoder = preproc["encoder"]
scaler: StandardScaler = preproc["scaler"]
num_classes = len(encoder.classes_)

model = MLP(INPUT_DIM, HIDDEN_DIMS, num_classes, DROPOUT).to(device)
model.load_state_dict(checkpoint)
model.eval()

def predict_location(image_file):
    embedding = image_to_clip_embedding(image_file).reshape(1, -1)
    X_scaled = scaler.transform(embedding)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    topk = probs.argsort()[::-1][:5]
    top_labels = encoder.inverse_transform(topk)

    result = {
        "probabilities": {label: float(probs[i]) for label, i in zip(top_labels, topk)},
        "topCountries": [
            {
                "name": label,
                "lat": 0.0,   # TODO: echte Koordinaten
                "lon": 0.0,   # TODO: echte Koordinaten
                "probability": float(probs[i])
            }
            for label, i in zip(top_labels, topk)
        ]
    }
    return result
