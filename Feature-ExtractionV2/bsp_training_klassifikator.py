import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

"""
Mit diesem Workflow kannst du deine Embeddings als Features verwenden und beliebige
Modelle (Logistic Regression, KNN, MLP etc.) darauf trainieren.
"""

# Embeddings und Metadaten laden
X = np.load("osv5m_clip_embeddings.npy")
meta = pd.read_csv("osv5m_clip_metadata.csv")

# Zielvariable wählen (z.B. Land)
y = meta["country"]

# Train/Validation split (hier aus dem gesamten Set, alternativ könntest du 'split' nutzen)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Klassifikator trainieren
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# Evaluierung
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
