from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

# Ordner, in dem die GeoJSON-Datei gespeichert ist
GEOJSON_FOLDER = 'path/to/your/backend/folder'

@app.route('/geojson')
def get_geojson():
    return send_from_directory(GEOJSON_FOLDER, 'geo.json')

if __name__ == '__main__':
    app.run(debug=True)
