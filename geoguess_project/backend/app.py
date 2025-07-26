from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Placeholder ML-Function
def predict_location(image_file):
    # Placeholder probablities
    probabilities = {
        'Deutschland': 0.6,
        'Frankreich': 0.15,
        'Italien': 0.1,
        'Spanien': 0.05,
        'Mexiko': 0.1
    }
    # Placeholder top countries and their probability
    top_countries = [
        {'name': 'Deutschland', 'lat': 51.1657, 'lon': 10.4515, 'probability': 0.6},
        {'name': 'Frankreich', 'lat': 46.6034, 'lon': 1.8883, 'probability': 0.15},
        {'name': 'Italien', 'lat': 41.8719, 'lon': 12.5674, 'probability': 0.1},
        {'name': 'Spanien', 'lat': 40.4637, 'lon': -3.7492, 'probability': 0.05},
        {'name': 'Mexiko', 'lat': 23.6345, 'lon': -102.5528, 'probability': 0.1}
    ]
    return {
        'probabilities': probabilities,
        'topCountries': top_countries
    }


# Toggle variant: REMOVE LATER
toggle = True

def get_dummy_data_1():
    probabilities = {
        'Deutschland': 0.6,
        'Frankreich': 0.15,
        'Italien': 0.1,
        'Spanien': 0.05,
        'Mexiko': 0.1
    }
    top_countries = [
        {'name': 'Deutschland', 'lat': 51.1657, 'lon': 10.4515, 'probability': 0.6},
        {'name': 'Frankreich', 'lat': 46.6034, 'lon': 1.8883, 'probability': 0.15},
        {'name': 'Italien', 'lat': 41.8719, 'lon': 12.5674, 'probability': 0.1},
        {'name': 'Spanien', 'lat': 40.4637, 'lon': -3.7492, 'probability': 0.05},
        {'name': 'Mexiko', 'lat': 23.6345, 'lon': -102.5528, 'probability': 0.1}
    ]
    return {'probabilities': probabilities, 'topCountries': top_countries}

def get_dummy_data_2():
    probabilities = {
        'Kanada': 0.5,
        'USA': 0.3,
        'Brasilien': 0.15,
        'Argentinien': 0.05
    }
    top_countries = [
        {'name': 'Kanada', 'lat': 56.1304, 'lon': -106.3468, 'probability': 0.5},
        {'name': 'USA', 'lat': 37.0902, 'lon': -95.7129, 'probability': 0.3},
        {'name': 'Brasilien', 'lat': -14.2350, 'lon': -51.9253, 'probability': 0.15},
        {'name': 'Argentinien', 'lat': -38.4161, 'lon': -63.6167, 'probability': 0.05}
    ]
    return {'probabilities': probabilities, 'topCountries': top_countries}
# REMOVE LATER


@app.route('/upload', methods=['POST'])
def upload():
    global toggle  # Global scope, REMOVE LATER
    if 'image' not in request.files:
        return jsonify({'error': 'Kein Bild hochgeladen!'}), 400
    
    image = request.files['image']
    # Save image if necessary:
    # image.save(f"./uploads/{image.filename}")

    # Call ML-Model here:
    results = predict_location(image)

    # TESTING ONLY: REMOVE LATER
    # Toggle between 2 dummy datasets
    if toggle:
        results = get_dummy_data_1()
    else:
        results = get_dummy_data_2()
    toggle = not toggle  # Toggle
    
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')