# DDoS Attack Mapping in 3d Globe Project

Visualize global DDoS attacks in real time using Cloudflare Radar data and machine learning anomaly scoring.  
Attacks are shown as animated arcs on a 3D globe in your browser.

---

## Features

- **Live DDoS attack visualization** using Cloudflare Radar API
- **Animated arcs** between origin and target countries
- **Severity scoring** using Isolation Forest anomaly detection
- **Interactive controls**: threshold, auto-rotate, labels
- **Test arc injection** for demo/testing
- **WebSocket live updates**

---

## Demo

![Screenshot](screenshot.png)

---

## Quick Start

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/ddos-globe-project.git
cd ddos-globe-project
```

### 2. Install Python dependencies

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Get a Cloudflare Radar API Token

- Go to [Cloudflare Radar](https://radar.cloudflare.com/)
- Create an API token with "Radar:Read" permission
- Save your token in a `.env` file:

```
CF_API_TOKEN=your_cloudflare_token_here
```

### 4. Run the backend server

```sh
uvicorn main:app --reload
```

By default, the server runs at [http://localhost:8000](http://localhost:8000).

### 5. Open the frontend

Visit [http://localhost:8000/app/](http://localhost:8000/app/) in your browser.

---

## File Structure

```
ddos_globe_project/
├── main.py                  # FastAPI backend, ML scoring, Cloudflare polling
├── country_centroids.json   # Country code → lat/lng mapping (all ISO codes)
├── static/
│   └── index.html           # Frontend: Globe.gl visualization
├── requirements.txt         # Python dependencies
├── .env                     # Your Cloudflare API token
```

---

## Configuration

- **Threshold slider**: Filter arcs by severity (move left to show more).
- **Auto-rotate**: Toggle globe rotation.
- **Labels**: Show/hide arc info.
- **Add Test Arc**: Inject a sample arc for demo/testing.

---

## Backend Details

- **Cloudflare Radar API**: Polled every 30 seconds for top origin/target countries.
- **ML Model**: Isolation Forest scores anomaly/severity for each attack.
- **Country centroids**: Uses `country_centroids.json` for accurate arc placement.
- **WebSocket**: Pushes live events to the frontend.

---

## Frontend Details

- **Globe.gl**: Renders animated arcs on a 3D globe.
- **REST API**: Loads initial events from `/api/events`.
- **WebSocket**: Receives live updates from `/ws`.
- **Responsive UI**: Works on desktop and mobile.

---

## Troubleshooting

- **No arcs shown?**
  - Lower the threshold slider.
  - Make sure your backend is running and reachable.
  - Check your `.env` for a valid Cloudflare API token.
  - Ensure `country_centroids.json` contains all ISO country codes.

- **API errors?**
  - Check backend logs for Cloudflare API errors.
  - Ensure your token has "Radar:Read" permission.

- **WebSocket not connecting?**
  - Make sure you access the frontend via the backend server (not just opening `index.html` directly).

---

## License

MIT License

---

## Credits

- [Cloudflare Radar](https://radar.cloudflare.com/)
- [Globe.gl](https://github.com/vasturiano/globe.gl)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Isolation Forest (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

---

## Contributing

Pull requests and issues welcome!
