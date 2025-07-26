from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import predict_location  # ← dein echtes Modell

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
