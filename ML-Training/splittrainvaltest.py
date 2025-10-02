import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

emb = np.load("embeddings_1-50.npy")
df  = pd.read_csv("metadata_1-50.csv")
labels = df["country"].values


emb_trainval, emb_test, y_trainval, y_test = train_test_split(
    emb, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
emb_train, emb_val, y_train, y_val = train_test_split(
    emb_trainval, y_trainval,
    test_size=0.125,
    stratify=y_trainval,
    random_state=42
)

print(f"Shapes → Train: {emb_train.shape}, Val: {emb_val.shape}, Test: {emb_test.shape}")
import numpy as np
import pandas as pd

np.save("emb_train.npy", emb_train)
np.save("emb_val.npy",   emb_val)
np.save("emb_test.npy",  emb_test)

pd.DataFrame({"country": y_train}).to_csv("y_train.csv", index=False)
pd.DataFrame({"country": y_val}).to_csv("y_val.csv", index=False)
pd.DataFrame({"country": y_test}).to_csv("y_test.csv", index=False)