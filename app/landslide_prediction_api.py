import os
from flask import Flask, request, jsonify
from app.model_utils import (
    load_models, preprocess_rgb_image, get_static_map_image,
    analyze_risk, generate_recommendations, generate_risk_map_image
)
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the ensemble models (U-Net and DeepLabV3+)
unet_model, deeplab_model = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    if 'location' not in request.form:
        return jsonify({"error": "Location parameter is required"}), 400

    location_name = request.form['location']

    # Fetch static map image for the location
    img, lat, lon = get_static_map_image(location_name)
    if img is None:
        return jsonify({"error": "Could not find location or fetch satellite data"}), 400

    # Preprocess the image for the model
    try:
        X = preprocess_rgb_image(img)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    # Make prediction using ensemble
    pred1 = unet_model.predict(X)
    pred2 = deeplab_model.predict(X)
    ensemble_pred = (pred1 + pred2) / 2.0

    # Analyze risk
    risk_map = analyze_risk(ensemble_pred[0, ..., 0])
    recommendations = generate_recommendations(risk_map)
    risk_map_img_b64 = generate_risk_map_image(risk_map)

    # Prepare response
    response = {
        "location": location_name,
        "lat": lat,
        "lon": lon,
        "prediction": float(np.mean(ensemble_pred)),
        "risk_map_image_base64": risk_map_img_b64,
        "risk_analysis": {
            "low_risk": float(np.mean(ensemble_pred < 0.3)),
            "medium_risk": float(np.mean((ensemble_pred >= 0.3) & (ensemble_pred < 0.7))),
            "high_risk": float(np.mean(ensemble_pred >= 0.7))
        },
        "recommendations": recommendations
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
