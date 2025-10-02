from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from predictor import predict_location 
import os
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'Kein Bild hochgeladen!'}), 400
    
    image = request.files['image']

    try:
        results = predict_location(image) 
        return jsonify(results) 
    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        traceback.print_exc() 
        print("-------------------------")
        return jsonify({'error': str(e)}), 500


@app.route('/geojson', methods=['GET'])
def geojson():
    try:
        return send_file('geo.json', mimetype='application/json')
    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        traceback.print_exc()  
        print("-------------------------")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
