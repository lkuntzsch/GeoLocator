from datasets import load_dataset
import shutil
import csv

# Metadaten laden ---------------------------------------------------------------
ds = load_dataset("osv5m/osv5m", full=False, trust_remote_code=True)
# ’full=False’ lädt Metadaten, aber keine Bilder

# Filter EU-Länder --------------------------------------------------------------
eu = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", 
      "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", 
      "RO", "SK", "SI", "ES", "SE"]
#eu2 = ["BE", "DE", "ES" ]

#ds = load_dataset("osv5m/osv5m", full=False, split="train")

ds_eu = ds.filter(lambda x: x["country"] in eu)
print(f"EU-Bilder: {len(ds_eu)}")



# Download ----------------------------------------------------------------------
def download_eu_images(row, hf_root="datasets/osv5m"):
    img_src = os.path.join(hf_root, row["img_path"])
    img_dst = os.path.join("your_eu_subset", row["img_path"])
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    shutil.copyfile(img_src, img_dst)

ds_eu.map(download_eu_images)


# Labels in CSV speichern für Zugriff
with open("metadata/eu_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img_path", "country", "latitude", "longitude"])
    for row in ds_eu:
        writer.writerow([row["img_path"], row["country"], row["latitude"], row["longitude"]])

