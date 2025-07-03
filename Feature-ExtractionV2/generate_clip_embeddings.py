import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


"""
Python-Skript zur Erzeugung von CLIP-Embeddings für das OSV5M-Dataset.

Voraussetzungen:
- Das Dataset liegt in zwei Ordnern: 'train' und 'test'.
- In jedem Ordner befindet sich eine Datei 'metadata.csv' mit Spalten:
  'filename', 'country', 'continent', 'osm_highway'.

Das Skript lädt CLIP, erstellt Batch-Embeddings und speichert:
- eine NumPy-Datei 'osv5m_clip_embeddings.npy' mit allen Embeddings
- eine CSV-Datei 'osv5m_clip_metadata.csv' mit Zuordnung von Index zu Labels
"""

"""
Dieses Skript liest alle Bilder aus den Ordnern train und test ein, berechnet für jedes mittels CLIP ein 512-dimensionales Embedding
und speichert:

    - osv5m_clip_embeddings.npy
    Ein NumPy-Array der Form (N, 512), wobei N die Gesamtzahl aller Bilder ist.

    - osv5m_clip_metadata.csv
    Eine Tabelle mit Spalten
        - index: Zeilenindex, der auf die Position im Embeddings-Array verweist
        - split: "train" oder "test"
        - filename, country, continent, osm_highway


"""

def generate_embeddings(data_dir, metadata_csv, model, processor, device, batch_size=64):
    df = pd.read_csv(metadata_csv)
    embeddings = []
    records = []
    for start in tqdm(range(0, len(df), batch_size), desc=f"Processing {os.path.basename(data_dir)}"):
        batch_df = df.iloc[start:start + batch_size]
        images = [
            Image.open(os.path.join(data_dir, row['filename'])).convert('RGB')
            for _, row in batch_df.iterrows()
        ]
        inputs = processor(images=images, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)
        image_embeds = image_embeds.cpu().numpy()
        embeddings.append(image_embeds)
        for idx, row in enumerate(batch_df.itertuples(index=False), start=start):
            records.append({
                'index': idx,
                'split': os.path.basename(data_dir),
                'filename': row.filename,
                'country': getattr(row, 'country', ''),
                'continent': getattr(row, 'continent', ''),
                'osm_highway': getattr(row, 'osm_highway', '')
            })
    return np.vstack(embeddings), pd.DataFrame(records)

def main():
    base_dir = "path_to_dataset"  # Pfad anpassen: Ordner mit 'train' und 'test'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    splits = ['train', 'test']
    all_embs = []
    all_recs = []
    for split in splits:
        data_dir = os.path.join(base_dir, split)
        metadata_csv = os.path.join(data_dir, "metadata.csv")
        embs, recs = generate_embeddings(data_dir, metadata_csv, model, processor, device)
        all_embs.append(embs)
        all_recs.append(recs)

    embeddings = np.vstack(all_embs)
    df = pd.concat(all_recs, ignore_index=True)

    np.save("osv5m_clip_embeddings.npy", embeddings)
    df.to_csv("osv5m_clip_metadata.csv", index=False)
    print("Fertig! 'osv5m_clip_embeddings.npy' und 'osv5m_clip_metadata.csv' wurden erstellt.")

if __name__ == "__main__":
    main()


