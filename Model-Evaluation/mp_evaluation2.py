import matplotlib.pyplot as plt
import pandas as pd
# === CLASSIFICATION REPORT DATA ===
classes = ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","GB","GR","HR","HU","IE","IT","LT","LU","LV","MT","NL","NO","PL","PT","RO","RS","SE","SI","SK"]

baseline_precision = [0.56,0.66,0.81,0.65,0.62,0.55,0.50,0.78,0.71,0.69,0.72,0.80,0.75,0.63,0.53,0.69,0.63,0.83,0.82,0.56,0.94,0.59,0.69,0.65,0.58,0.82,0.69,0.81,0.67,0.62]
baseline_recall    = [0.51,0.63,0.83,0.63,0.58,0.46,0.67,0.76,0.66,0.85,0.51,0.76,0.80,0.52,0.63,0.87,0.59,0.76,0.88,0.75,0.98,0.77,0.77,0.59,0.75,0.80,0.74,0.67,0.72,0.65]
baseline_f1        = [0.53,0.65,0.82,0.64,0.60,0.50,0.57,0.77,0.68,0.76,0.59,0.78,0.77,0.57,0.58,0.77,0.61,0.79,0.85,0.64,0.96,0.67,0.73,0.62,0.66,0.81,0.71,0.73,0.69,0.63]

tuned_precision = [0.59,0.71,0.89,0.73,0.72,0.63,0.61,0.86,0.79,0.80,0.76,0.83,0.82,0.74,0.68,0.79,0.72,0.88,0.90,0.70,0.96,0.69,0.78,0.74,0.71,0.86,0.82,0.87,0.73,0.74]
tuned_recall    = [0.68,0.73,0.89,0.73,0.71,0.55,0.74,0.84,0.73,0.88,0.61,0.84,0.86,0.65,0.73,0.89,0.67,0.84,0.92,0.81,0.99,0.80,0.83,0.71,0.81,0.86,0.83,0.76,0.79,0.76]
tuned_f1        = [0.63,0.72,0.89,0.73,0.72,0.59,0.67,0.85,0.76,0.84,0.67,0.83,0.84,0.69,0.70,0.84,0.69,0.86,0.91,0.75,0.97,0.74,0.80,0.72,0.75,0.86,0.83,0.81,0.76,0.75]

"""
# === Plot F1-Vergleich pro Klasse ===
x = range(len(classes))
plt.figure(figsize=(14,6))
plt.bar(x, baseline_f1,  color = "#1e34dc", width=0.4, label='Baseline F1')
plt.bar([i+0.4 for i in x], tuned_f1, color= "#24e6e9",  width=0.4, label='Tuned F1')
plt.xticks([i+0.2 for i in x], classes, rotation=90)
plt.ylabel('F1-Score')
plt.title('F1-Score pro Klasse: Baseline vs. Tuned')
plt.legend()
plt.tight_layout()
plt.show()

# Precision- und Recall-Vergleiche
for metric_name, base_vals, tuned_vals in [("Precision", baseline_precision, tuned_precision), ("Recall", baseline_recall, tuned_recall)]:
    plt.figure(figsize=(14,6))
    plt.bar(x, base_vals, color = "#1e34dc", width=0.4, label=f'Baseline {metric_name}')
    plt.bar([i+0.4 for i in x], tuned_vals, color= "#24e6e9", width=0.4, label=f'Tuned {metric_name}')
    plt.xticks([i+0.2 for i in x], classes, rotation=90)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} pro Klasse: Baseline vs. Tuned')
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
# === Precision-Recall-Balkendiagramme für ausgewählte Klassen ===
important_classes = ["DE", "FR", "IT", "GB", "ES"]

indices = [classes.index(c) for c in important_classes]

for metric_name, values in [("Precision", tuned_precision), ("Recall", tuned_recall)]:
    plt.figure(figsize=(8,5))
    plt.bar(important_classes, [values[i] for i in indices], color='skyblue')
    plt.ylim(0,1)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} für ausgewählte Klassen (Tuned Model)')
    plt.show()
