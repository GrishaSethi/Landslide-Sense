from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from typing import Optional
import os
from contextlib import asynccontextmanager

from app.model_utils import load_models, predict_landslide, get_static_map_image, preprocess_rgb_image, generate_risk_map_image

# Lifespan event handler for FastAPI >=0.104
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    zoom: Optional[int] = 16

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Determine lat/lon from input
        if request.location_name:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="landslide_prediction")
            location = geolocator.geocode(request.location_name)
            if not location:
                raise HTTPException(status_code=404, detail="Could not geocode location name.")
            lat, lon = location.latitude, location.longitude
        elif request.lat is not None and request.lon is not None:
            lat, lon = request.lat, request.lon
        else:
            raise HTTPException(status_code=400, detail="Must provide either lat/lon or location_name.")

        # Fetch satellite image using OSM static map
        img, _, _ = get_static_map_image(f"{lat},{lon}", zoom=request.zoom, size=(128,128))
        if img is None:
            raise HTTPException(status_code=404, detail="Could not fetch satellite image for the location.")
        X = preprocess_rgb_image(img)
        prediction, risk_map, risk_level = predict_landslide(X, app.state.models)
        risk_map_b64 = generate_risk_map_image(risk_map)
        landslide_present = 'yes' if prediction == 1 else 'no'
        return {
            "landslide_present": landslide_present,
            "risk_map_image": risk_map_b64,
            "risk_level": risk_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
