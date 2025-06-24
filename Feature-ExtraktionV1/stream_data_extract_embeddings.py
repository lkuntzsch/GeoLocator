import random
import os
import numpy as np
import torch
import clip
from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from huggingface_hub import login

# Konfiguration
#EU_COUNTRIES = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
#                "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
#                "RO", "SK", "SI", "ES", "SE"]
EU_COUNTRIES = ["AT", "BE", "BG", "HR"]
SAMPLES_PER_COUNTRY = 500 # maybe später auf 1000 hochsetzen 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "embeddings"

# Ordner erstellen
#os.makedirs(SAVE_DIR, exist_ok=True)

# token hugging face weil gesperrt wegen rate limit
login("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")

# Modell laden
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Streaming Dataset laden
# -> Streaming statt laden der metasaten um speicher zu sparen
ds = load_dataset("osv5m/osv5m", split="train", streaming=True, trust_remote_code=True)

# Samples puffern (zwischenspeichern)
# sammelt begrenzte Anzahl samples (entsprechend Variable)
sampled = defaultdict(list)
max_per_country = {c: SAMPLES_PER_COUNTRY for c in EU_COUNTRIES}
buffer = []

print("Sammle Beispiele pro Land...")
for example in tqdm(ds):
    country = example["country"]
    if country in EU_COUNTRIES:
        save_path = os.path.join(SAVE_DIR, f"{country}.npz")
        if os.path.exists(save_path):
            continue # wenn bereits verarbeitet

        if len(sampled[country]) < SAMPLES_PER_COUNTRY:
            sampled[country].append(example)


        if all(
        len(v) >= SAMPLES_PER_COUNTRY or os.path.exists(os.path.join(SAVE_DIR, f"{c}.npz"))
        for c, v in sampled.items()
        ):
            break

""" Sieht nun etwa so aus:
sampled = {
  "DE": [img1, img2, ..., img1000],
  "FR": [img1, img2, ..., img1000],
  ...
}

"""

# Flatten + mischen - doch nicht hier weil pro land um zwischenerg. zu speichern
#samples = [img for country_imgs in sampled.values() for img in country_imgs]
#random.shuffle(samples)

"""
Flatten = zusammenführen zu Liste mit 500 x N Bildern, die Reihenfolge ist länderweise gruppiert

Mischen = random shuffle um Verarbeitung gleichmäßig zu verteilen, Bias vermeiden
-> gerade bei wenig Daten sinnvoll, später für training hilfreich
"""

print(" Bilder werden verarbeitet und die Embeddings erstellt...")

# Embeddings erzeugen
# V2 mit Batches, weil zu lange braucht sonst
BATCH_SIZE = 16 # wenn gpu mach 32

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ----------------------------------------------------------------------
# Pro Land verarbeiten und speichern
for country, examples in sampled.items():
    save_path = os.path.join(SAVE_DIR, f"{country}.npz")
    if os.path.exists(save_path):
        print(f"✅ {country}: bereits verarbeitet, wird übersprungen.")
        continue
    print(f"\n Verarbeite {country} ({len(examples)} Bilder)...")
    random.shuffle(examples)

    embeddings = []
    labels = []

    for batch in tqdm(batchify(examples, BATCH_SIZE), desc=f"{country} Batches"):
        batch_images = []
        batch_labels = []

        for ex in batch:
            try:
                img = Image.open(ex["image"]).convert("RGB")
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                batch_labels.append(EU_COUNTRIES.index(ex["country"]))
            except Exception as e:
                print(f"ACHTUNG! Fehler bei Bild: {e}")
                continue

        if not batch_images:
            continue

        image_input = torch.stack(batch_images).to(DEVICE)

        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.extend(features.cpu().numpy())
            labels.extend(batch_labels)

    # Speichern
    if embeddings:
        np.savez_compressed(save_path,
                            embeddings=np.array(embeddings),
                            labels=np.array(labels))
        print(f"{country}: {len(labels)} Embeddings gespeichert in {save_path}")
    else:
        print(f"ACHTUNG!! {country}: Keine gültigen Bilder, nichts gespeichert.")


"""
Erstellt 1 npz file für *alle* Embeddings
-> einfaches Laden für Training und Visualisierung
-> spart Speicher, keine Dateiverwaltung
-> blöd wenn Länder nachladen

Alternative wäre ein npy oder npz pro Land
-> gut zum debuggen pro land
-> komplexeres Training, weil alles zusammengeführt wird
-> macht sinn bei größeren Datensätzen sagt chat?
"""

""" URSPRÜNGLICHER CODE mit einem npz file für alle - führte zu too many requests
error in hugging face:
(davor noch image_embeddings und labels unter batch_size definieren)


for batch in tqdm(batchify(samples, BATCH_SIZE)): #v2 diese schleife
    batch_images = []
    batch_labels = []

    for ex in batch: #v1 tqdm(samples):
        try:
            img = Image.open(ex["image"]).convert("RGB")
            #v1 image_input = preprocess(img).unsqueeze(0).to(DEVICE)
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
            batch_labels.append(EU_COUNTRIES.index(ex["country"]))

            #with torch.no_grad():
            #    features = model.encode_image(image_input)
            #    features = features / features.norm(dim=-1, keepdim=True)
            #    image_embeddings.append(features.cpu().numpy()[0])
            #    labels.append(EU_COUNTRIES.index(ex["country"]))  # integer label
        except Exception as e:
            print("Fehler bei Bild:", e)
            continue
    
    if not batch_images:
        continue  # wenn alle Bilder im Batch fehlschlagen

    image_input = torch.stack(batch_images).to(DEVICE)

    with torch.no_grad():
        features = model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        image_embeddings.extend(features.cpu().numpy())
        labels.extend(batch_labels)    

# Als .npz speichern
np.savez_compressed(SAVE_PATH,
                    embeddings=np.array(image_embeddings),
                    labels=np.array(labels))
print(f"{len(labels)} Embeddings gespeichert unter {SAVE_PATH}")

"""