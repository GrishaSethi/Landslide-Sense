# Landslide Sense 🌍

Satellite Imagery–Driven Deep Learning for Landslide Risk Assessment

This project provides a backend service for predicting landslide susceptibility using satellite images and deep learning segmentation models (U-Net and DeepLabV3+). The system runs as a Flask REST API that accepts a location as input and returns a risk map, severity score, analysis, and actionable recommendations.

## 🚀 Features

- **Deep-learning–based landslide risk segmentation**
- **Supported Models:**
  - U-Net
  - DeepLabV3+
- **Comprehensive Output:**
  - Risk classification (Low/Medium/High)
  - Segmented heatmap (.png visualization)
  - Analytical insights and severity assessment
  - Actionable mitigation recommendations
- **Modular Design** for future integration with satellite APIs (Google Earth Engine, Sentinel Hub, etc.)
- **RESTful API** for easy integration with frontend applications

## 📦 Installation & Setup

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

Follow the preprocessing format outlined in `preprocess_image()` within `app/model_utils.py` or use the provided sample dataset.

### 3️⃣ Train Models

```bash
python train_models.py
```

This will generate the trained models in the `models/` directory:

```
models/
│── landslide_unet_model.h5
└── landslide_deeplab_model.h5
```

### 4️⃣ Start the API

```bash
python app/landslide_prediction_api.py
```

The API will be available at `http://localhost:5000`

## 🔌 API Usage

### Endpoint

```
POST /predict
```

### Form Data

| Key | Type | Description |
|-----|------|-------------|
| location | string | Location name (static dataset used in demo) |

### Sample Request

```bash
curl -X POST -F "location=Darjeeling" http://localhost:5000/predict
```

### Sample Response

```json
{
  "location": "Darjeeling",
  "risk_level": "High",
  "risk_map_path": "generated/risk_map_83c2.png",
  "analysis": "High soil moisture and slope instability detected in marked zones.",
  "recommendations": [
    "Avoid heavy construction in red zones",
    "Monitor land movement during monsoon season",
    "Install surface drainage systems"
  ]
}
```

## 📂 Project Structure

```
/Landslide-Sense/
│
├── app/
│   ├── landslide_prediction_api.py      # Flask REST API
│   ├── model_utils.py                   # Preprocessing & model loading
│   ├── data_generator.py                # Data pipeline utilities
│   └── utils.py                         # Helper methods
│
├── models/
│   ├── landslide_unet_model.h5
│   └── landslide_deeplab_model.h5
│
├── sample_data/
│   └── sample_landslide_data.h5
│
├── generated/                           # Output directory for risk maps
│   └── risk_map_*.png
│
├── requirements.txt
├── train_models.py
└── README.md
```

## 🛠 Technical Details

### Model Architecture

- **U-Net:** Encoder-decoder architecture with skip connections for precise segmentation
- **DeepLabV3+:** Atrous spatial pyramid pooling for multi-scale feature extraction

### Input Data

- Satellite imagery with multiple spectral bands
- Topographical data (elevation, slope)
- Geological and soil composition data
- Real-time satellite data integration

### Output

- Binary segmentation masks identifying landslide-prone areas
- Risk heatmaps with probability scores
- Comprehensive risk assessment reports

## ⚠ Important Notes

- The current demo uses a static HDF5 dataset for all location requests
- For real-world deployment, integrate dynamic satellite imagery retrieval from services like Google Earth Engine or Sentinel Hub
- Ensure your dataset follows the format required in `preprocess_image()` function
- Model performance may vary based on regional geological characteristics

## 🔮 Future Enhancements

- Multi-temporal analysis for change detection
- Ensemble modeling for improved accuracy
- Mobile application integration
- Historical landslide database correlation

---

**Landslide Sense** - Empowering communities with AI-driven landslide risk assessment 🌍🛡
