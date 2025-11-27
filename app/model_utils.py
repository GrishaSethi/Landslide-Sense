import os
import numpy as np
import h5py
from geopy.geocoders import Nominatim
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@register_keras_serializable()
def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@register_keras_serializable()
def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UNET_PATH = os.path.join(PROJECT_ROOT, 'landslide_unet_model.h5')
DEEPLAB_PATH = os.path.join(PROJECT_ROOT, 'landslide_deeplab_model.h5')
CUSTOM_OBJECTS = {
    'dice_loss': dice_loss,
    'dice_coefficient': dice_coefficient,
    'precision_m': precision_m,
    'recall_m': recall_m,
    'f1_m': f1_m
}

import joblib
ENSEMBLE_PATH = os.path.join(PROJECT_ROOT, 'landslide_ensemble.pkl')

def load_models():
    unet = load_model(UNET_PATH, custom_objects=CUSTOM_OBJECTS)
    deeplab = load_model(DEEPLAB_PATH, custom_objects=CUSTOM_OBJECTS)
    ensemble = joblib.load(ENSEMBLE_PATH)
    return {'unet': unet, 'deeplab': deeplab, 'ensemble': ensemble}

def preprocess_image(image_path):
    with h5py.File(image_path, 'r') as hdf:
        data = np.array(hdf.get('img'), dtype=np.float32)
        data[np.isnan(data)] = 0.000001
        data_red = data[:, :, 3] / 255.0
        data_green = data[:, :, 2] / 255.0
        data_blue = data[:, :, 1] / 255.0
        data_nir = data[:, :, 7] / 255.0
        data_slope = data[:, :, 12] / 90.0
        data_elevation = data[:, :, 13] / 5000.0
        data_ndvi = np.divide(data_nir - data_red, data_nir + data_red + 1e-6)
        X = np.zeros((1, 128, 128, 6), dtype=np.float32)
        X[0, :, :, 0] = data_red
        X[0, :, :, 1] = data_green
        X[0, :, :, 2] = data_blue
        X[0, :, :, 3] = data_ndvi
        X[0, :, :, 4] = data_slope
        X[0, :, :, 5] = data_elevation
    return X

import requests
from PIL import Image
import io
import base64

def get_static_map_image(location_name, zoom=15, size=(128,128)):
    geolocator = Nominatim(user_agent="landslide_prediction")
    location = geolocator.geocode(location_name)
    if not location:
        return None, None, None
    lat, lon = location.latitude, location.longitude
    # OpenStreetMap static map (no API key needed)
    url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z={zoom}&size={size[0]},{size[1]}&l=sat"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None, lat, lon
    img = Image.open(io.BytesIO(resp.content)).convert('RGB').resize(size)
    return img, lat, lon

def predict_landslide(X, models):
    # X: (1, 128, 128, 6)
    pred_unet = models['unet'].predict(X)
    pred_deeplab = models['deeplab'].predict(X)
    # Use mean of mask as feature for ensemble (or flatten, depending on training)
    features = np.array([
        np.mean(pred_unet),
        np.mean(pred_deeplab)
    ]).reshape(1, -1)
    pred_ensemble = models['ensemble'].predict(features)
    risk_level = 'High' if pred_ensemble[0] == 1 else 'Low'
    # For visualization, use Deeplab output as risk map
    risk_map = (pred_deeplab[0, :, :, 0] > 0.5).astype(np.uint8)
    return int(pred_ensemble[0]), risk_map, risk_level

def preprocess_rgb_image(img):
    # img: PIL Image (128x128 RGB)
    arr = np.array(img).astype(np.float32) / 255.0
    # If model expects 6 channels, pad with zeros for demo
    X = np.zeros((1, 128, 128, 6), dtype=np.float32)
    X[0, :, :, 0:3] = arr / 1.0  # R,G,B
    # Leave NDVI, slope, elevation as zeros
    return X

def generate_risk_map_image(risk_map):
    # risk_map: 2D array of 1 (green), 2 (yellow), 3 (red)
    color_map = {1: [0,255,0], 2: [255,255,0], 3: [255,0,0]}
    h, w = risk_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_map.items():
        rgb[risk_map == k] = v
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

def analyze_risk(prediction):
    risk_map = np.zeros_like(prediction)
    risk_map[prediction < 0.3] = 1
    risk_map[(prediction >= 0.3) & (prediction < 0.7)] = 2
    risk_map[prediction >= 0.7] = 3
    return risk_map

def generate_recommendations(risk_map):
    red_zones = np.sum(risk_map == 3)
    yellow_zones = np.sum(risk_map == 2)
    recommendations = []
    if red_zones > 0:
        recommendations.append("High risk areas detected. Immediate evacuation recommended.")
        recommendations.append("Consider installing ground sensors for monitoring.")
        recommendations.append("Consult with geotechnical engineers for stabilization.")
    if yellow_zones > 0:
        recommendations.append("Medium risk areas detected. Regular monitoring recommended.")
        recommendations.append("Consider vegetation planting to stabilize slopes.")
    if len(recommendations) == 0:
        recommendations.append("No significant landslide risk detected. Regular monitoring still advised.")
    return recommendations
