import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Klassenliste
classes = [
    "AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI",
    "FR","GB","GR","HR","HU","IE","IT","LT","LU","LV",
    "MT","NL","NO","PL","PT","RO","RS","SE","SI","SK"
]

# --- Baseline Model Test-Set Metriken ---
baseline_precision = [
    0.51,0.61,0.79,0.69,0.63,0.44,0.55,0.75,0.67,0.74,
    0.60,0.72,0.81,0.71,0.63,0.82,0.65,0.82,0.81,0.70,
    0.93,0.72,0.85,0.60,0.76,0.85,0.78,0.81,0.72,0.75
]
baseline_recall = [
    0.56,0.60,0.87,0.62,0.55,0.53,0.61,0.83,0.77,0.84,
    0.55,0.85,0.80,0.56,0.62,0.81,0.63,0.80,0.86,0.69,
    0.97,0.64,0.70,0.66,0.64,0.81,0.74,0.73,0.68,0.56
]
baseline_f1 = [
    0.53,0.60,0.83,0.65,0.59,0.48,0.58,0.79,0.72,0.79,
    0.57,0.78,0.80,0.63,0.62,0.82,0.64,0.81,0.83,0.69,
    0.95,0.67,0.77,0.63,0.69,0.83,0.76,0.77,0.70,0.64
]

# --- Tuned Model Test-Set Metriken ---
tuned_precision = [
    0.52,0.61,0.80,0.70,0.65,0.46,0.56,0.75,0.68,0.75,
    0.61,0.72,0.81,0.72,0.64,0.82,0.66,0.83,0.82,0.71,
    0.93,0.73,0.85,0.61,0.77,0.86,0.79,0.82,0.73,0.76
]
tuned_recall = [
    0.58,0.61,0.88,0.63,0.56,0.54,0.62,0.84,0.78,0.84,
    0.56,0.86,0.81,0.58,0.63,0.82,0.64,0.81,0.86,0.70,
    0.97,0.65,0.71,0.68,0.65,0.82,0.75,0.73,0.69,0.57
]
tuned_f1 = [
    0.55,0.61,0.84,0.66,0.60,0.49,0.59,0.79,0.73,0.80,
    0.58,0.79,0.81,0.64,0.63,0.82,0.65,0.82,0.84,0.70,
    0.95,0.68,0.77,0.64,0.70,0.84,0.77,0.77,0.71,0.65
]
# === PLOTS ===
plt.style.use('seaborn-v0_8-colorblind')  # or try 'ggplot', 'seaborn-dark', etc.

# Custom color palette
colors = {
    'Baseline Train Loss': "#1e34dc",
    'Baseline Val Loss': "#fdd210",
    'Tuned Train Loss': "#24e6e9",
    'Tuned Val Loss': "#e7610e",
    'Baseline Train Acc': '#1e34dc',
    'Baseline Val Acc': '#fdd210',
    'Tuned Train Acc': '#24e6e9',
    'Tuned Val Acc': '#e7610e'
}

# --- Gesamtmetriken ---
baseline_metrics = {"Accuracy": 0.69, "Top-5 Accuracy": 0.8823, "Log-Loss": 2.4210}
tuned_metrics    = {"Accuracy": 0.70, "Top-5 Accuracy": 0.8996, "Log-Loss": 1.8960}

# --- Precision, Recall, F1 Vergleich pro Klasse ---
x = np.arange(len(classes))
width = 0.35

# Precision
plt.figure(figsize=(18,5))
plt.bar(x - width/2, baseline_precision, width, color = "#1e34dc", label='Baseline Precision', alpha=0.7)
plt.bar(x + width/2, tuned_precision,    width, label='Tuned Precision', color= "#24e6e9",   alpha=0.7)
plt.xticks(x, classes, rotation=90)
plt.ylabel('Precision')
plt.title('Precision pro Klasse – Baseline vs. Tuned (KNN)')
plt.legend()
plt.tight_layout()
plt.show()

# Recall
plt.figure(figsize=(18,5))
plt.bar(x - width/2, baseline_recall, width, label='Baseline Recall',  color = "#1e34dc", alpha=0.7)
plt.bar(x + width/2, tuned_recall,    width, label='Tuned Recall',  color= "#24e6e9",  alpha=0.7)
plt.xticks(x, classes, rotation=90)
plt.ylabel('Recall')
plt.title('Recall pro Klasse – Baseline vs. Tuned (KNN)')
plt.legend()
plt.tight_layout()
plt.show()

# F1-Score
plt.figure(figsize=(18,5))
plt.bar(x - width/2, baseline_f1, width, label='Baseline F1',  color = "#1e34dc", alpha=0.7)
plt.bar(x + width/2, tuned_f1,    width, label='Tuned F1', color= "#24e6e9",    alpha=0.7)
plt.xticks(x, classes, rotation=90)
plt.ylabel('F1-Score')
plt.title('F1-Score pro Klasse – Baseline vs. Tuned (KNN)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Vergleich Overall-Metriken ---
metrics_df = pd.DataFrame({
    'Metric': list(baseline_metrics.keys()),
    'Baseline': list(baseline_metrics.values()),
    'Tuned': list(tuned_metrics.values())
})

plt.figure(figsize=(8,5))
plt.bar(metrics_df['Metric'], metrics_df['Baseline'], width=0.4, label='Baseline', alpha=0.7)
plt.bar(metrics_df['Metric'], metrics_df['Tuned'],    width=0.4, label='Tuned',    alpha=0.7)
plt.ylabel('Wert')
plt.title('Vergleich Gesamt-Metriken – Baseline vs. Tuned (KNN)')
plt.legend()
plt.tight_layout()
plt.show()