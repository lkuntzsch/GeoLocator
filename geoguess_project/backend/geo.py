from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

# Den Ordner, in dem sich geo.json befindet (das gleiche Verzeichnis wie app.py)
GEOJSON_FOLDER = os.getcwd()  # Aktuelles Verzeichnis, in dem sich app.py befindet

@app.route('/geojson')
def get_geojson():
    try:
        # Sende die geo.json Datei zurück, die im gleichen Verzeichnis wie app.py liegt
        return send_from_directory(GEOJSON_FOLDER, 'geo.json')
    except FileNotFoundError:
        return jsonify({'error': 'GeoJSON-Datei nicht gefunden'}), 404

if __name__ == '__main__':
    app.run(debug=True)
