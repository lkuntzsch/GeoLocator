# GeoLocator
BV2 Projekt SS25
## Ablauf
1. Bilder mit Geotags sammeln (GeoYFCC, Flickr, Wikimedia)
2. Pro Bild: Land/Region als Label extrahieren
3. Für jedes Bild: CLIP-Embedding berechnen  
4. Embeddings + Labels → Klassifikationsmodell (LogReg, kNN, MLP)
5. Bei Nutzeranfrage: Embedding vom Bild → Modellvorhersage + ähnliche Trainingsbilder zeigen


### **Modulare Aufteilung**
| Aufgabe | Titel                                                      | Personen |
| ------- | ---------------------------------------------------------- | -------- |
| 1       | **CLIP-basierte Feature-Extraktion & Embedding-Datenbank** | A + B    |
| 2       | **Klassifikator (z. B. k-NN, LogReg) & Inferenz-Service**  | A + C    |
| 3       | **Web-App (Frontend + API-Endpunkte)**                     | B + D    |
| 4       | **Erklärbarkeit: Ähnliche Bilder & Kartenanzeige**         | C + D    |

#### 🔹 Aufgabe 1: Feature-Extraktion mit CLIP & Embedding-Datenbank (A + B)

> Extrahiere Embeddings für einen Open-Source-Datensatz (z. B. GeoYFCC, Flickr, Wikimedia). Organisiere sie in einer Embedding-Datenbank mit Länderlabels.

**Teilaufgaben:**

- Datensatz vorbereiten: Bild-URLs, Geotags, Labels (Land, Region)
    
- `clip-vit-base` oder `ViT-B/32` verwenden → 512-dim Embedding
    
- Embeddings serialisieren (z. B. als `.npy` + JSON-Labels oder mit Faiss)
    
- Optional: PCA/UMAP zum Plotten/Verstehen
    

**Tools:** `torch`, `CLIP`, `numpy`, `scikit-learn`, evtl. `faiss`


#### 🔹 Aufgabe 2: Klassifikation + Inferenz-Service (A + C)

> Trainiere ein Modell (z. B. Logistic Regression oder k-NN), das für ein neues Bild den wahrscheinlichsten Aufnahmeort vorhersagt.

**Teilaufgaben:**

- Train/Val-Split auf Embedding-Level
    
- Klassifikator trainieren (Top-1 Accuracy, Confusion Matrix, Top-k Evaluation)
    
- Deployment als REST-Endpunkt (FastAPI): `/predict` → Bild → Vorhersage (Land + Score)
    

**Tools:** `scikit-learn`, `FastAPI`, `joblib`, `CLIP`, `PIL`


#### 🔹 Aufgabe 3: Web-App & Upload-Handling (B + D)

> Frontend zur Bildauswahl, Upload und Ergebnisanzeige entwickeln.

**Teilaufgaben:**

- UI mit Bild-Upload, Vorschau, "Absenden"-Button
    
- REST-Anfrage an Backend (`/predict`)
    
- Anzeige: Wahrscheinlichstes Land + Text + Karte
    

**Tools:** `React`, `Tailwind`, `axios`, evtl. `leaflet` für die Karte

#### 🔹 Aufgabe 4: Ähnliche Bilder & Kartendarstellung (C + D)

> Zeige dem Nutzer, warum das Modell diesen Ort vorhersagt. Dazu: ähnlichste Bilder aus Trainingsdaten + Confidence + Karte.

**Teilaufgaben:**

- Berechne Embedding-Distanz (cosine oder L2) zu Trainingsdaten
    
- Zeige 3–5 ähnlichste Bilder (Visualisierung)
    
- Anzeige auf Karte (Land einfärben oder Flagge setzen)
    
- Textvorschlag: „Das Bild ähnelt typischen Aufnahmen aus … (97 % Sicherheit)“
    

**Tools:** `scikit-learn`, `OpenStreetMap`, `Leaflet`, `React`, ggf. `matplotlib` für Debug


#### **Möglicher Wochenplan**
| Woche | Aufgabe 1 (A+B)                        | Aufgabe 2 (A+C)               | Aufgabe 3 (B+D)           | Aufgabe 4 (C+D)                          |
| ----- | -------------------------------------- | ----------------------------- | ------------------------- | ---------------------------------------- |
| 1     | Setup, CLIP testen, Mini-Dataset bauen | Modellskizze, Metriken klären | Upload-Prototyp           | Karten- und Ähnlichkeitsidee ausarbeiten |
| 2     | Embedding-Pipeline automatisieren      | Training, Val-Metriken testen | UI Upload + API-Call      | Karte + Ähnliche Bilder planen           |
| 3     | Embedding-Datenbank aufbauen           | API-Endpunkt bereitstellen    | Upload-Endpunkt anbinden  | Ähnliche-Bild-Funktion                   |
| 4     | Batch-Processing optimieren            | Live-Inferenz                 | Ergebnisanzeige           | Kartenintegration                        |
| 5     | Finalisierung Datenstruktur            | Feinschliff + Top-k Anzeige   | UI-Styling, Ladeindikator | Konfidenz-Text, Bildvergleich UI         |
| 6     | Backup + Docs                          | Evaluation + Vergleich        | Feedback & Polishing      | Deployment & Präsentation                |



