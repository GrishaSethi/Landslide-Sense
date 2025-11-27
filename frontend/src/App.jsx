import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { predictLandslide } from './api';

function LocationMarker({ setLatLon }) {
  useMapEvents({
    click(e) {
      setLatLon([e.latlng.lat, e.latlng.lng]);
    },
  });
  return null;
}

function App() {
  const [latLon, setLatLon] = useState([27.7, 85.3]); // Default: Kathmandu
  const [zoom, setZoom] = useState(16);
  const [locationName, setLocationName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await predictLandslide(latLon[0], latLon[1], zoom);
      setResult(data);
    } catch (e) {
      setError("Prediction failed: " + (e?.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const handlePredictByLocation = async () => {
    if (!locationName.trim()) {
      setError("Please enter a location name.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location_name: locationName, zoom }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        if (errJson.detail && errJson.detail.toLowerCase().includes('not found')) {
          setResult({ landslide_present: 'no', risk_level: 'Low', risk_map_image: null });
          setLoading(false);
          return;
        }
        throw new Error(errJson.detail || "API error");
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError("Prediction failed: " + (e.message || e));
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 24 }}>
      <h1>Landslide Risk Mapping</h1>
      <div style={{ height: 350, marginBottom: 16 }}>
        <MapContainer center={latLon} zoom={zoom} style={{ height: '100%', width: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <Marker position={latLon} />
          <LocationMarker setLatLon={setLatLon} />
        </MapContainer>
      </div>
      <div style={{ marginBottom: 12 }}>
        <label>Latitude: <input type="number" value={latLon[0]} step="0.0001" onChange={e => setLatLon([parseFloat(e.target.value), latLon[1]])} /></label>
        <label style={{ marginLeft: 8 }}>Longitude: <input type="number" value={latLon[1]} step="0.0001" onChange={e => setLatLon([latLon[0], parseFloat(e.target.value)])} /></label>
        <label style={{ marginLeft: 8 }}>Zoom: <input type="number" value={zoom} min={1} max={19} onChange={e => setZoom(parseInt(e.target.value))} /></label>
        <button style={{ marginLeft: 16 }} onClick={handlePredict} disabled={loading}>{loading ? 'Predicting...' : 'Predict'}</button>
      </div>
      <div style={{ marginBottom: 12 }}>
        <label>Or enter location name: <input type="text" value={locationName} onChange={e => setLocationName(e.target.value)} placeholder="e.g. Kathmandu" style={{ width: 200 }}/></label>
        <button style={{ marginLeft: 8 }} onClick={handlePredictByLocation} disabled={loading}>{loading ? 'Predicting...' : 'Predict by Name'}</button>
      </div>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 24 }}>
          <h2>Prediction Result</h2>
          {result.landslide_present && (
            <div>Landslide present: <b>{result.landslide_present}</b></div>
          )}
          {result.risk_level && (
            <div>Risk Level: <b>{result.risk_level}</b></div>
          )}
          {result.risk_map_image && (
            <div style={{marginTop: 16}}>
              <div>Risk Map:</div>
              <img src={`data:image/png;base64,${result.risk_map_image}`} alt="Risk Map" style={{width: 256, height: 256, border: '1px solid #ccc'}} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App
