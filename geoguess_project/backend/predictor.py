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

country_coords = {
    "DE": (51.1657, 10.4515),
    "GB": (55.3781, -3.4360),
    "FR": (46.6034, 1.8883),
    "SE": (60.1282, 18.6435),
    "PL": (51.9194, 19.1451),
    "IT": (41.8719, 12.5674),
    "ES": (40.4637, -3.7492),
    "RO": (45.9432, 24.9668),
    "PT": (39.3999, -8.2245),
    "NL": (52.1326, 5.2913),
    "FI": (61.9241, 25.7482),
    "DK": (56.2639, 9.5018),
    "GR": (39.0742, 21.8243),
    "HU": (47.1625, 19.5033),
    "AT": (47.5162, 14.5501),
    "IE": (53.1424, -7.6921),
    "NO": (60.4720, 8.4689),
    "EE": (58.5953, 25.0136),
    "LV": (56.8796, 24.6032),
    "BE": (50.5039, 4.4699),
    "HR": (45.1000, 15.2000),
    "CZ": (49.8175, 15.4730),
    "RS": (44.0165, 21.0059),
    "CH": (46.8182, 8.2275),
    "SK": (48.6690, 19.6990),
    "BG": (42.7339, 25.4858),
    "LT": (55.1694, 23.8813),
    "SI": (46.1512, 14.9955),
    "LU": (49.8153, 6.1296),
    "MT": (35.9375, 14.3754)
}

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
                "lat": country_coords.get(label, (0.0, 0.0))[0],
                "lon": country_coords.get(label, (0.0, 0.0))[1],
                "probability": float(probs[i])
            }
            for label, i in zip(top_labels, topk)
        ]
    }
    return result
