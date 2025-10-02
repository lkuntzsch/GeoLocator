import numpy as np
import os

X, y = [], []
for file in os.listdir("embeddings"):
    data = np.load(os.path.join("embeddings", file))
    X.append(data["embeddings"])
    y.append(data["labels"])

X = np.concatenate(X)
y = np.concatenate(y)


# flatten und mischen?