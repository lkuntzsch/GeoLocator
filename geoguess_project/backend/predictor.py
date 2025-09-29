# backend/predictor.py

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ml_model import MLP
from clip_utils import image_to_clip_embedding
import pycountry
import requests
from functools import lru_cache

# Modellparameter
HIDDEN_DIMS = [1024, 512]
DROPOUT     = 0.21911440122299433  # falls du das exakt übernehmen willst
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

# Helper: ISO‑Code → Wikipedia‑Seitentitel
# Manuelle Zuordnung für deutsche Wikipedia-Titel
ISO_TO_WIKI_DE = {
    "PL": "Polen",
    "DE": "Deutschland",
    "FR": "Frankreich",
    "IT": "Italien",
    "ES": "Spanien",
    "SE": "Schweden",
    "FI": "Finnland",
    "PT": "Portugal",
    "NL": "Niederlande",
    "GR": "Griechenland",
    "AT": "Österreich",
    "DK": "Dänemark",
    "BE": "Belgien",
    "CZ": "Tschechien",
    "HU": "Ungarn",
    "RO": "Rumänien",
    "BG": "Bulgarien",
    "HR": "Kroatien",
    "SK": "Slowakei",
    "SI": "Slowenien",
    "LT": "Litauen",
    "LV": "Lettland",
    "EE": "Estland",
    "IE": "Irland",
    "LU": "Luxemburg",
    "NO": "Norwegen",
    "CH": "Schweiz",
    "GB": "Vereinigtes_Königreich",
    "MT": "Malta"
}

def iso_to_wiki_title(code: str, lang: str = 'de') -> str:
    code = code.upper()
    if lang == 'de' and code in ISO_TO_WIKI_DE:
        return ISO_TO_WIKI_DE[code].replace(" ", "_")
    
    country = pycountry.countries.get(alpha_2=code)
    if not country:
        return code
    return country.name.replace(' ', '_')



# Helper: Wikipedia‑API mit Caching
# In predictor.py

@lru_cache(maxsize=128)
def get_wikipedia_info(title: str, lang: str = 'de'):
    S = requests.Session()

    # 1. Define a User-Agent to identify your application
    headers = {
        'User-Agent': 'SpotiFindApp/1.0 (MyGeolocatorProject; contact@example.com)'
    }
    # 2. Add the headers to your request session
    S.headers.update(headers)

    URL = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop":    "extracts|pageimages",
        "titles":  title,
        "exintro": True,
        "explaintext": True,
        "piprop":  "thumbnail",
        "pithumbsize": 300,
        "format":  "json"
    }

    # 3. Make the request
    response = S.get(URL, params=params)
    
    # 4. (Optional but good practice) Check if the request was successful
    response.raise_for_status()

    # 5. Parse the JSON response
    resp = response.json()

    pages = resp.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    thumb   = page.get("thumbnail", {}).get("source", "")
    return {"description": extract, "thumbnail": thumb}

def predict_location(image_file):
    # CLIP-Embedding extrahieren
    embedding = image_to_clip_embedding(image_file).reshape(1, -1)
    # Skalierung
    X_scaled = scaler.transform(embedding)
    # In Tensor umwandeln und auf das Device schieben
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Inferenz
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Top‑5 Länder ermitteln
    topk = probs.argsort()[::-1][:5]
    top_labels = encoder.inverse_transform(topk)

    # Ergebnisse aufbereiten, inkl. Wikipedia‑Info
    top_countries = []
    for label, i in zip(top_labels, topk):
        lat, lon = country_coords.get(label, (0.0, 0.0))
        wiki_title = iso_to_wiki_title(label)
        wiki = get_wikipedia_info(wiki_title)
        top_countries.append({
            "code":        label,
            "lat":         lat,
            "lon":         lon,
            "probability": float(probs[i]),
            "wiki": {
                "description": wiki["description"],
                "thumbnail":   wiki["thumbnail"],
                "source": f"https://de.wikipedia.org/wiki/{wiki_title}"
            }
        })

    return {
        "probabilities": {
            label: float(probs[i]) for label, i in zip(top_labels, topk)
        },
        "topCountries": top_countries,
        "relevantCountries": [country['code'] for country in top_countries] 
    }
