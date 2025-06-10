import torch
import clip
from PIL import Image

# Gerät wählen: CUDA (GPU) oder CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modell und Preprocessing laden
model, preprocess = clip.load("ViT-B/32", device=device)

# Bild laden und vorverarbeiten
image = preprocess(Image.open("images/australia-nsw-blue-mountains.jpg")).unsqueeze(0).to(device)

# Keine Gradienten berechnen
with torch.no_grad():
    image_features = model.encode_image(image)

# Optional: Embedding normalisieren
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Ergebnis ausgeben
print("Bild-Embedding (shape):", image_features.shape)
print("Erstes Embedding:", image_features[0][:])  # erste 10 Werte anzeigen


#Für Batch-Verarbeitung kannst du einfach mehrere Bilder in einer Liste verarbeiten und dann torch.stack verwenden