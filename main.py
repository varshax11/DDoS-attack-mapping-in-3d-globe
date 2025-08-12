import asyncio, os, math, time, json
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import uvicorn
from dotenv import load_dotenv
load_dotenv()
import pycountry

with open("country_centroids.json") as f:
    CC2LL = json.load(f)
    for cc, entry in CC2LL.items():
        if "lon" in entry:
            entry["lng"] = entry["lon"]
        print(f"Loaded centroid: {cc} -> ({entry['lat']}, {entry['lng']})")

# -----------------------------
# Config
# -----------------------------
CF_API_TOKEN = os.getenv("CF_API_TOKEN")
if not CF_API_TOKEN:
    raise SystemExit("Set CF_API_TOKEN env var (Cloudflare Radar -> Read token).")

# Updated API endpoints based on current Cloudflare Radar API structure
CF_BASE = "https://api.cloudflare.com/client/v4/radar"
CF_L3_TOP_ORIGINS = f"{CF_BASE}/attacks/layer3/top/locations/origin"
CF_L3_TOP_TARGETS = f"{CF_BASE}/attacks/layer3/top/locations/target"

POLL_SECONDS = 30
ROLL_LIMIT = 800       # how many recent events to keep for snapshot
TRAIN_WINDOW = 600     # how many feature rows to keep for the model
MIN_TRAIN = 120        # minimum rows needed before fitting model

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="DDoS Globe (Cloudflare + ML Severity)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

app.mount("/app", StaticFiles(directory="static", html=True), name="app")

EVENTS_BUFFER = deque(maxlen=ROLL_LIMIT)

# -----------------------------
# Utils
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def pick_latlng(cc: Optional[str]) -> Optional[Tuple[float,float]]:
    if not cc: return None
    entry = CC2LL.get(cc.upper())
    if not entry:
        print(f"DEBUG: CC2LL missing {cc.upper()}")
    if entry and "lat" in entry and "lng" in entry:
        return (entry["lat"], entry["lng"])
    return None

def extract_cc_from_location(location_data: dict) -> Optional[str]:
    """Extract country code from Cloudflare location data structure."""
    # Try different possible structures in Cloudflare's response
    if isinstance(location_data, str) and 2 <= len(location_data) <= 3:
        return location_data.upper()
    
    if isinstance(location_data, dict):
        # Try various keys that might contain the country code
        for key in ['code', 'country_code', 'iso_code', 'alpha2', 'cc']:
            if key in location_data:
                val = location_data[key]
                if isinstance(val, str) and 2 <= len(val) <= 3:
                    return val.upper()
    
    return None

def get_date_range_params() -> Dict[str, str]:
    """Generate date parameters for the last 7 days in the correct ISO format."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    return {
        "dateStart": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateEnd": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

# -----------------------------
# ML Anomaly Model (rolling Isolation Forest)
# Features:
#   f0: pct (0..100 from Radar "top" endpoint)
#   f1: pct - EMA_pair
#   f2: pct - EMA_origin
#   f3: pct - EMA_target
#   f4: sin(hour*2π/24)
#   f5: cos(hour*2π/24)
#
# We maintain EMAs per (origin→target) pair, per origin, per target.
# Every poll, we add new rows and (re)fit IsolationForest on last TRAIN_WINDOW rows.
# -----------------------------
class Ema:
    def __init__(self, alpha: float = 0.2):
        self.mu: Optional[float] = None
        self.alpha = alpha
    def update(self, x: float) -> float:
        if self.mu is None:
            self.mu = x
        else:
            self.mu = self.alpha * x + (1 - self.alpha) * self.mu
        return self.mu
    def value(self) -> Optional[float]:
        return self.mu

class AnomalyModel:
    def __init__(self):
        self.X_buf: deque = deque(maxlen=TRAIN_WINDOW)
        self._scaler = StandardScaler()
        self._iforest: Optional[IsolationForest] = None
        # baselines
        self.ema_pair: Dict[str, Ema] = defaultdict(Ema)
        self.ema_origin: Dict[str, Ema] = defaultdict(Ema)
        self.ema_target: Dict[str, Ema] = defaultdict(Ema)
        # for normalizing raw anomaly scores to 0..1
        self.raw_scores: deque = deque(maxlen=TRAIN_WINDOW)

    def _features(self, o_cc: str, t_cc: str, pct: float, ts: datetime) -> np.ndarray:
        key_pair = f"{o_cc}->{t_cc}"
        mu_pair = self.ema_pair[key_pair].update(pct)
        mu_ori  = self.ema_origin[o_cc].update(pct)
        mu_tar  = self.ema_target[t_cc].update(pct)
        # deltas (if EMA not ready, treat delta as 0)
        d_pair = pct - (mu_pair if mu_pair is not None else pct)
        d_ori  = pct - (mu_ori  if mu_ori  is not None else pct)
        d_tar  = pct - (mu_tar  if mu_tar  is not None else pct)
        hour = ts.hour + ts.minute/60.0
        ang = 2*math.pi*hour/24.0
        return np.array([pct, d_pair, d_ori, d_tar, math.sin(ang), math.cos(ang)], dtype=float)

    def add_and_score(self, o_cc: str, t_cc: str, pct: float, ts: datetime) -> Tuple[float, float, int]:
        """
        Returns (pct_norm, anomaly_norm, severity_int).
        """
        x = self._features(o_cc, t_cc, pct, ts)
        self.X_buf.append(x)

        n = len(self.X_buf)
        # Train (or re-train) if we have enough data
        if n >= MIN_TRAIN:
            X = np.vstack(self.X_buf)
            Xs = self._scaler.fit_transform(X)
            self._iforest = IsolationForest(
                n_estimators=200, contamination=0.05, random_state=42
            )
            self._iforest.fit(Xs)

            # score the latest sample
            xs = self._scaler.transform(x.reshape(1, -1))
            # IsolationForest: larger decision_function => more normal.
            raw = float(-self._iforest.decision_function(xs)[0])  # positive larger => more anomalous
            self.raw_scores.append(raw)
            # min-max normalize anomaly in window
            rmin = min(self.raw_scores) if self.raw_scores else raw
            rmax = max(self.raw_scores) if self.raw_scores else raw
            anomaly01 = 0.0 if rmax == rmin else (raw - rmin) / (rmax - rmin)
        else:
            anomaly01 = 0.0  # cold start until we have enough samples

        pct01 = clamp01(pct / 100.0)

        # Blend into severity (0..100). Weights: 45% raw %, 55% anomaly.
        sev = int(round(100.0 * (0.45 * pct01 + 0.55 * anomaly01)))
        sev = max(1, min(100, sev))
        return pct01, anomaly01, sev

anom = AnomalyModel()

# -----------------------------
# WebSocket connections
# -----------------------------
class ConnectionManager:
    def __init__(self):
        self._clients: set[WebSocket] = set()
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)
    async def broadcast(self, msg: Dict[str, Any]):
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# -----------------------------
# FIXED: Cloudflare Radar polling with correct response parsing
# -----------------------------
def parse_cloudflare_response(response_data: dict) -> List[dict]:
    """
    Parse the Cloudflare Radar API response based on the actual structure.
    Based on logs showing keys like ['top_0', 'meta'], this handles the current format.
    """
    items = []
    
    if not isinstance(response_data, dict):
        print("Response is not a dict")
        return items
    
    # Handle different response structures
    if "result" in response_data:
        result = response_data["result"]
        print(f"Processing result with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            # Look for 'top_X' keys (like 'top_0' seen in logs)
            top_keys_found = []
            for key, value in result.items():
                if key.startswith("top_"):
                    top_keys_found.append(key)
                    if isinstance(value, list):
                        print(f"Found {len(value)} items in {key}")
                        items.extend(value)
                    else:
                        print(f"Key {key} is not a list, it's a {type(value).__name__}")
            
            if top_keys_found:
                print(f"Processed top_X keys: {top_keys_found}")
            else:
                print("No top_X keys found")
            
            # Fallback: check for other common structures
            if not items:
                print("No items found in top_X keys, trying fallbacks...")
                if isinstance(result, list):
                    items = result
                    print(f"Used result as direct list: {len(items)} items")
                elif "data" in result and isinstance(result["data"], list):
                    items = result["data"]
                    print(f"Used result.data: {len(items)} items")
                elif "top" in result and isinstance(result["top"], list):
                    items = result["top"]
                    print(f"Used result.top: {len(items)} items")
                else:
                    print("No fallback structures found")
    else:
        print("No 'result' key in response")
    
    print(f"Total items extracted: {len(items)}")
    return items

def extract_location_and_percentage(item: dict) -> Optional[Tuple[str, float]]:
    """
    Extract country code and percentage from Cloudflare API response item.
    Updated to handle the actual Cloudflare response format.
    """
    if not isinstance(item, dict):
        return None
    
    # Extract percentage/value first
    pct = 0.0
    pct_found = False
    
    # Try "value" first (this is what Cloudflare returns)
    if "value" in item:
        try:
            pct = float(item["value"])
            pct_found = True
        except (ValueError, TypeError):
            pass
    
    # Fallback to other possible keys
    if not pct_found:
        for pct_key in ["percentage", "pct", "count", "bytes", "requests"]:
            if pct_key in item:
                try:
                    pct = float(item[pct_key])
                    pct_found = True
                    break
                except (ValueError, TypeError):
                    continue
    
    if not pct_found:
        print(f"No percentage found in item: {item}")
        return None
    
    # Extract country code - updated for actual Cloudflare format
    cc = None
    
    # Check for origin/target specific fields first
    for cc_key in [
        "originCountryAlpha2",    # For origins endpoint
        "targetCountryAlpha2",    # For targets endpoint  
        "countryAlpha2",          # Generic
        "alpha2",                 # Alternative
        "code",                   # Generic code
        "country_code",           # Alternative format
        "location",               # Fallback
        "country"                 # Generic country
    ]:
        if cc_key in item:
            cc_candidate = item[cc_key]
            if isinstance(cc_candidate, str) and 2 <= len(cc_candidate) <= 3:
                cc = cc_candidate.upper()
                break
            elif isinstance(cc_candidate, dict):
                # If it's a nested object, try to extract from it
                nested_cc = extract_cc_from_location(cc_candidate)
                if nested_cc:
                    cc = nested_cc
                    break
    
    if not cc:
        print(f"No country code found in item: {item}")
        return None
    
    print(f"Extracted: {cc} = {pct}%")
    return (cc, pct)

async def _fetch_locations(client: httpx.AsyncClient, endpoint: str, location_type: str) -> list[dict]:
    """Fetch top locations from Cloudflare Radar API with proper error handling."""
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    # Updated parameters based on current API requirements
    params = {
        "limit": 40,
        "format": "json",
        **get_date_range_params()  # Add date range parameters
    }
    
    try:
        print(f"Fetching {location_type} from {endpoint}")
        print(f"Params: {params}")
        
        r = await client.get(endpoint, headers=headers, params=params, timeout=30)
        
        # Print response details for debugging
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Response headers: {dict(r.headers)}")
            print(f"Response text: {r.text[:1000]}")  # First 1000 chars
            r.raise_for_status()
        
        j = r.json()
        print(f"Response structure: {list(j.keys()) if isinstance(j, dict) else 'Not a dict'}")
        
        # Use the new parsing function immediately
        items = parse_cloudflare_response(j)
        print(f"Final: Parsed {len(items)} items from {location_type}")
        
        # Debug: print sample parsed item
        if items:
            print(f"Sample parsed item: {json.dumps(items[0], indent=2)}")
        else:
            print("No items found - this could be normal if there are no attacks in the time period")
        
        return items
            
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP error while fetching {location_type}: {e}")
        print(f"[ERROR] Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching {location_type}: {e}")
        return []

# COMPLETELY REWRITTEN: This is the key fix!
async def fetch_top_locations_both(client: httpx.AsyncClient) -> list[dict]:
    """Fetch both origin and target locations concurrently and create meaningful arcs."""
    try:
        # Fetch origin and target locations concurrently
        origin_task = _fetch_locations(client, CF_L3_TOP_ORIGINS, "origins")
        target_task = _fetch_locations(client, CF_L3_TOP_TARGETS, "targets")
        
        origin_items, target_items = await asyncio.gather(origin_task, target_task)
        
        print(f"Got {len(origin_items)} origins, {len(target_items)} targets")
    except Exception as e:
        print(f"[ERROR] Error during concurrent fetch of origins/targets: {e}")
        origin_items, target_items = [], []

    now_dt = datetime.now(timezone.utc)
    events: list[dict] = []

    # Extract valid country data
    origin_countries = []
    for item in origin_items:
        location_data = extract_location_and_percentage(item)
        if location_data:
            origin_countries.append(location_data)
    
    target_countries = []
    for item in target_items:
        location_data = extract_location_and_percentage(item)
        if location_data:
            target_countries.append(location_data)
    
    print(f"Extracted {len(origin_countries)} origin countries, {len(target_countries)} target countries")

    # STRATEGY 1: Direct pairing (origin[0] -> target[0], origin[1] -> target[1], etc.)
    max_pairs = min(len(origin_countries), len(target_countries), 15)
    for i in range(max_pairs):
        o_cc, o_pct = origin_countries[i]
        t_cc, t_pct = target_countries[i]
        
        # Skip if same country (no visible arc)
        if o_cc == t_cc:
            print(f"Skipping same-country pair: {o_cc}")
            continue
            
        o_ll = pick_latlng(o_cc)
        t_ll = pick_latlng(t_cc)
        
        if not (o_ll and t_ll):
            print(f"Missing coordinates for {o_cc} -> {t_cc}")
            continue
        
        # Use average percentage for arc severity
        avg_pct = (o_pct + t_pct) / 2.0
        _, _, severity = anom.add_and_score(o_cc, t_cc, avg_pct, now_dt)
        
        events.append({
            "id": f"direct_{time.time_ns()}_{i}",
            "time": now_dt.isoformat(),
            "type": f"L3 DDoS ({o_cc}→{t_cc})",
            "gbps": round(avg_pct * 0.5, 1),  # Fake Gbps based on percentage
            "pps": None,
            "severity": severity,
            "origin": {"cc": o_cc, "lat": o_ll[0], "lng": o_ll[1]},
            "target": {"cc": t_cc, "lat": t_ll[0], "lng": t_ll[1]},
        })

    # STRATEGY 2: Cross-matrix (each origin -> multiple targets)
    # Take top 5 origins and top 8 targets, create multiple arcs
    for i, (o_cc, o_pct) in enumerate(origin_countries[:5]):
        o_ll = pick_latlng(o_cc)
        if not o_ll:
            continue
            
        # Each origin attacks 2-3 different targets
        target_count = 0
        for j, (t_cc, t_pct) in enumerate(target_countries[:8]):
            if target_count >= 3:  # Limit arcs per origin
                break
            if o_cc == t_cc:  # Skip same country
                continue
                
            t_ll = pick_latlng(t_cc)
            if not t_ll:
                continue
            
            # Weight severity: origin activity (70%) + target activity (30%)
            combined_pct = (o_pct * 0.7) + (t_pct * 0.3)
            _, _, severity = anom.add_and_score(o_cc, t_cc, combined_pct, now_dt)
            
            events.append({
                "id": f"cross_{time.time_ns()}_{i}_{j}",
                "time": now_dt.isoformat(),
                "type": f"L3 DDoS ({o_cc}→{t_cc})",
                "gbps": round(combined_pct * 0.3, 1),
                "pps": None,
                "severity": severity,
                "origin": {"cc": o_cc, "lat": o_ll[0], "lng": o_ll[1]},
                "target": {"cc": t_cc, "lat": t_ll[0], "lng": t_ll[1]},
            })
            
            target_count += 1

    print(f"Generated {len(events)} meaningful arcs (origin≠target)")
    
    # Debug: show a few examples
    if events:
        for i, event in enumerate(events[:3]):
            print(f"Sample arc {i+1}: {event['origin']['cc']} -> {event['target']['cc']} (severity: {event['severity']})")
    
    return events

async def radar_poller():
    """Main polling loop with better error handling and backoff."""
    consecutive_errors = 0
    max_consecutive_errors = 10
    base_delay = POLL_SECONDS
    
    async with httpx.AsyncClient() as client:
        # Initial fetch
        try:
            print("Starting initial fetch...")
            initial = await fetch_top_locations_both(client)
            for e in initial:
                EVENTS_BUFFER.append(e)
            print(f"Initial fetch complete: {len(initial)} events")
            consecutive_errors = 0
        except Exception as e:
            print(f"[ERROR] Initial fetch failed: {e}")
            consecutive_errors += 1

        # Main polling loop
        while True:
            try:
                # Calculate delay with exponential backoff on errors
                delay = min(base_delay * (2 ** min(consecutive_errors, 5)), 300)  # Max 5 minutes
                await asyncio.sleep(delay)
                
                print("Fetching new data...")
                items = await fetch_top_locations_both(client)
                
                if items:  # Only process if we got data
                    for e in items:
                        e["time"] = now_iso()
                        EVENTS_BUFFER.append(e)
                        await manager.broadcast({"kind": "event", "item": e})
                    consecutive_errors = 0
                    print(f"Successfully processed {len(items)} events")
                else:
                    print("No events received, but no error occurred")
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Cloudflare fetch error (attempt {consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many consecutive errors ({consecutive_errors}), stopping poller")
                    break

# -----------------------------
# API
# -----------------------------
@app.get("/api/events")
async def latest_events(limit: int = 200):
    return JSONResponse(list(EVENTS_BUFFER)[-limit:])

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({"kind": "snapshot", "items": list(EVENTS_BUFFER)[-250:]})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.on_event("startup")
async def _startup():
    asyncio.create_task(radar_poller())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)