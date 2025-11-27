import axios from 'axios';

const API_BASE = 'http://localhost:8000'; // Adjust if backend runs elsewhere

export const predictLandslide = async (lat, lon, zoom = 16) => {
  const res = await axios.post(`${API_BASE}/predict`, { lat, lon, zoom });
  return res.data;
};
