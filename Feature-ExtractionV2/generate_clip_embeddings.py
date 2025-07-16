import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import argparse


"""
Python-Skript zur Erzeugung von CLIP-Embeddings für das OSV5M-Dataset.

Voraussetzungen:
- Das Dataset liegt nicht mehr in zwei Ordnern 'train' und 'test' sondern in einem Ordner 'images'
- In dem Ordner befindet sich eine Datei 'metadata_combinded.csv' 
  mit einer Spalte 'id', die dem Dateinamen <id>.jpg entspricht, sowie allen Metadaten (z.B. 'country', 'latitude', …).


Das Skript lädt CLIP, erstellt Batch-Embeddings und speichert:
- eine NumPy-Datei 'osv5m_clip_embeddings.npy' mit allen Embeddings
- eine CSV-Datei 'osv5m_clip_metadata.csv' mit Zuordnung von Index zu Labels und Metadaten
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


def generate_embeddings(data_dir, df_part, model, processor, device, batch_size=64):
    """
    data_dir   : Pfad zum images_part_XX Ordner
    df_part    : DataFrame mit den Zeilen der Metadaten, die wir in diesem Ordner tatsächlich verarbeiten
    """

    embeddings = []
    records = []

    # In Batches über das DataFrame iterieren
    for start in tqdm(range(0, len(df_part), batch_size), desc=f"Processing {os.path.basename(data_dir)}"):
        batch_df = df_part.iloc[start:start + batch_size]

        images, valid_indices = [], []

        for i, row in batch_df.iterrows():
            img_path = os.path.join(data_dir, f"{row['id']}.jpg")
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert('RGB'))
                valid_indices.append(i)
            else:
                tqdm.write(f"[WARN] Datei nicht gefunden: {img_path}")
        
        if not images:
            continue

        # CLIP-Inputs erzeugen und Embeddings berechnen
        inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs).cpu().numpy()
        embeddings.append(image_embeds)

        # Metadaten-Rekorde anlegen
        for j, idx in enumerate(valid_indices):
            rec = df_part.loc[idx].to_dict()  # ?enthält alle Spalten aus metadata_*.csv für partial df (5 ordner)?
            rec.update({
                'index': start+j,
                'split': os.path.basename(data_dir),
                'filename': f"{rec['id']}.jpg"
            })
            records.append(rec)

    # Embeddings als (N, D)-Array, Records als DataFrame
    if embeddings:
        return np.vstack(embeddings), pd.DataFrame(records)
    else:
        return np.empty((0, model.config.projection_dim)), pd.DataFrame(records)

def main():
    num_parts = 5 # ACHTUNG anpassen, wenn 5 zu viel sein sollten
    base_dir = "path_to_dataset"  # ACHTUNG -  Pfad anpassen: Ordner 'images'
    meta_path = "path_to_csv" # ACHTUNG - Pfad anpassen: Pfad zu 'metadata-combined.csv'

    # CLIP laden
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # parse wegen ordnern
    parser = argparse.ArgumentParser(
        description="Erzeuge CLIP-Embeddings in Chargen von images_part_XX Ordnern."
    )
    parser.add_argument("--start", type=int, default=1,
                        help="Erste Ordner-Nummer (01–50), z.B. 1 für images_part_01")
    parser.add_argument("--output_embeddings", required=True,
                        help="Dateiname für die NumPy-Ausgabe, z.B. osv5m_clip_embeddings_01-05.npy")
    parser.add_argument("--output_csv", required=True,
                        help="Dateiname für die Metadaten-Ausgabe, z.B. osv5m_clip_metadata_01-05.csv")
    args = parser.parse_args()

    # Metadaten einlesen
    df_meta = pd.read_csv(meta_path)

    all_embs = []
    all_recs = []

    # Ordner liste
    parts = [
        f"images_part_{i:02d}"
        for i in range(args.start, args.start + num_parts)
        if i >= 1 and i <= 50
    ]

    for part in parts:
        data_dir = os.path.join(base_dir, part)
        if not os.path.isdir(data_dir):
            print(f"[WARN] Ordner nicht gefunden, übersprungen: {data_dir}")
            continue

        # nur ids die in diesen orndern existieren
        vorhandene_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(data_dir)
            if f.lower().endswith(".jpg")
        }
        df_part = df_meta[df_meta['id'].astype(str).isin(vorhandene_ids)].reset_index(drop=True)

        if df_part.empty:
            print(f"[INFO] Keine passenden IDs in {part}, übersprungen.")
            continue


        embs, recs = generate_embeddings(data_dir, df_part, model, processor, device)
        all_embs.append(embs)
        all_recs.append(recs)

    # Gesamtdaten zusammenführen
    if all_embs:
        embeddings = np.vstack(all_embs)
        df_out = pd.concat(all_recs, ignore_index=True)

        # Ausgeben
        np.save(args.output_embeddings, embeddings)
        df_out.to_csv(args.output_csv, index=False) 
        print("Fertig! Embeddings: {args.output_embeddings}, Metadaten: {args.output_csv} wurden erstellt.")
     else:
        print("SOS! Keine Daten verarbeitet. Parameter prüfen!")

if __name__ == "__main__":
    main()


