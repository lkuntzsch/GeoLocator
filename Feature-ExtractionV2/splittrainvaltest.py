import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) bereits bereinigten Embeddings und Labels laden
emb = np.load("C:/Users/lisak/Documents/Studium/Master/Praktikum BV, ML, CV/Prak25/Embeddings/embeddings_all_eu.npy")
df = pd.read_csv("C:/Users/lisak/Documents/Studium/Master/Praktikum BV, ML, CV/Prak25/Embeddings/metadata_all_eu.csv.xls", sep=",", encoding="utf-8")
#df.to_csv("C:/Users/lisak/Documents/Studium/Master/Praktikum BV, ML, CV/Prak25/Embeddings/metadata_all_eu.csv.xls", sep=",", index=False, encoding="utf-8")
labels = df["country"].values

# (Optional: Filter z.B. only countries ≥ 10 samples, wie gehabt)

# 2) Erst-Teilung: Train+Val (80 %) vs. Test (20 %)
emb_trainval, emb_test, y_trainval, y_test = train_test_split(
    emb, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 3) Zweite Teilung: Train (≈70 % gesamt) vs. Val (≈10 % gesamt)
#    Hier nehmen wir 12.5 % von den 80 % für Val → 0.125
emb_train, emb_val, y_train, y_val = train_test_split(
    emb_trainval, y_trainval,
    test_size=0.125,
    stratify=y_trainval,
    random_state=42
)

print(f"Shapes → Train: {emb_train.shape}, Val: {emb_val.shape}, Test: {emb_test.shape}")
import numpy as np
import pandas as pd

# Embeddings speichern
np.save("emb_train.npy", emb_train)
np.save("emb_val.npy",   emb_val)
np.save("emb_test.npy",  emb_test)

# Labels als CSV speichern
pd.DataFrame({"country": y_train}).to_csv("y_train.csv", index=False)
pd.DataFrame({"country": y_val}).to_csv("y_val.csv", index=False)
pd.DataFrame({"country": y_test}).to_csv("y_test.csv", index=False)