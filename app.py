from datetime import datetime
import heapq
import io
import math
from pathlib import Path
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

st.set_page_config(
    page_title="ASTRA-SCAN | Chandrayaan South Pole Rover OS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 1. WORKSPACE PATHS & ROVER CONSTANTS
# ==========================================
WORKSPACE_DIR = Path(__file__).parent.resolve()
ISRO_DATA_DIR = WORKSPACE_DIR / "isro_data"
UNPACKED_DIR = WORKSPACE_DIR / "datasets" / "unpacked_archive"
VAULT_DIR = WORKSPACE_DIR / "lunar_southpole_vault"
MODEL_DIR = WORKSPACE_DIR / "models"
MODEL_PATH = MODEL_DIR / "southpole_lookahead_ai.pkl"
DB_PATH = str(WORKSPACE_DIR / "rover_mission.db")

for _d in (ISRO_DATA_DIR, UNPACKED_DIR, VAULT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 96
CELL_METERS = 10.0
PRAGYAN_MASS_KG = 26.0
SOLAR_PANEL_MAX_W = 50.0


# ==========================================
# 2. RELATIONAL MISSION DATABASE (SQLITE)
# ==========================================
class SouthPoleVault:
    def __init__(self, db_path: str = DB_PATH, vault_dir: Path = VAULT_DIR):
        self.db_path = db_path
        self.vault_dir = vault_dir
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self._seed_missions()

    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS missions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                agency TEXT DEFAULT 'ISRO',
                landing_site TEXT,
                status TEXT DEFAULT 'Active',
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS uploaded_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_format TEXT NOT NULL,
                instrument_name TEXT DEFAULT 'Unknown',
                pds4_lid TEXT DEFAULT 'N/A',
                upload_timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS simulation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                solar_elevation_deg REAL,
                rover_tilt_limit_deg REAL,
                battery_soc_pct REAL,
                net_energy_w REAL,
                explanation TEXT
            );

            CREATE TABLE IF NOT EXISTS mission_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                log_level TEXT NOT NULL,
                module TEXT NOT NULL,
                message TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS breadcrumb_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                epoch_sec REAL,
                grid_r INTEGER,
                grid_c INTEGER,
                lat REAL,
                lon REAL,
                signal_pct REAL,
                mode TEXT
            );

            CREATE TABLE IF NOT EXISTS southpole_media_vault (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_file TEXT,
                grid_r INTEGER,
                grid_c INTEGER,
                temp_k REAL,
                ai_prediction TEXT,
                synced_to_earth INTEGER
            );
            """
        )
        conn.commit()
        conn.close()

    def _seed_missions(self):
        conn = sqlite3.connect(self.db_path)
        missions = [
            ("CH-2", "Chandrayaan-2 Orbiter", "ISRO", "Lunar Polar Orbit", "Active", "TMC-2 DEM & CLASS X-Ray Spectra"),
            ("CH-3", "Chandrayaan-3 Vikram & Pragyan", "ISRO", "Shiv Shakti Point (69.373° S, 32.319° E)", "Completed", "ChaSTE Thermal, LIBS & NavCam Stereo"),
            ("ASTRA", "ASTRA-SCAN South Pole Rover", "ISRO / Astra-2026", "Shackleton-Shiv Shakti Corridor", "Active", "60m AI Lookahead & 10-Min SAFETY BACKUP"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO missions (code, name, agency, landing_site, status, description) VALUES (?, ?, ?, ?, ?, ?)",
            missions,
        )
        conn.commit()
        conn.close()

    def log_event(self, level: str, module: str, message: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO mission_logs (timestamp, log_level, module, message) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%H:%M:%S"), level, module, message),
        )
        conn.commit()
        conn.close()

    def register_dataset_file(self, mission_name: str, file_path: Path, instrument: str = "Unknown", pds4_lid: str = "N/A"):
        conn = sqlite3.connect(self.db_path)
        exists = conn.execute("SELECT id FROM uploaded_datasets WHERE file_path = ?", (str(file_path),)).fetchone()
        if not exists and file_path.exists():
            conn.execute(
                """
                INSERT INTO uploaded_datasets
                (mission_name, filename, file_path, file_size_bytes, file_format, instrument_name, pds4_lid, upload_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mission_name,
                    file_path.name,
                    str(file_path),
                    file_path.stat().st_size,
                    file_path.suffix.lower(),
                    instrument,
                    pds4_lid,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
        conn.close()

    def log_simulation(self, sun_deg: float, tilt_limit: float, battery_pct: float, net_w: float, note: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO simulation_history
            (timestamp, solar_elevation_deg, rover_tilt_limit_deg, battery_soc_pct, net_energy_w, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().strftime("%H:%M:%S"), sun_deg, tilt_limit, battery_pct, round(net_w, 2), note),
        )
        conn.commit()
        conn.close()

    def record_breadcrumb(self, r: int, c: int, lat: float, lon: float, signal_pct: float, mode: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO breadcrumb_trail
            (timestamp, epoch_sec, grid_r, grid_c, lat, lon, signal_pct, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%H:%M:%S"),
                time.time(),
                int(r),
                int(c),
                round(float(lat), 3),
                round(float(lon), 3),
                round(float(signal_pct), 1),
                mode,
            ),
        )
        conn.commit()
        conn.close()

    def save_southpole_image(self, img_data: Image.Image, r: int, c: int, temp_k: float, ai_pred: str, signal_ok: bool) -> str:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"SP_S{r:02d}_{c:02d}_{ts_tag}.png"
        filepath = self.vault_dir / filename
        img_data.save(filepath)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO southpole_media_vault
            (timestamp, image_file, grid_r, grid_c, temp_k, ai_prediction, synced_to_earth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%H:%M:%S"),
                str(filepath),
                int(r),
                int(c),
                round(float(temp_k), 1),
                ai_pred,
                1 if signal_ok else 0,
            ),
        )
        conn.commit()
        conn.close()
        self.log_event("INFO", "VAULT", f"Saved South Pole capture {filename} (synced={signal_ok})")
        return str(filepath)

    def flush_downlink_queue(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("UPDATE southpole_media_vault SET synced_to_earth = 1 WHERE synced_to_earth = 0")
        synced_count = cur.rowcount
        conn.commit()
        conn.close()
        if synced_count > 0:
            self.log_event("INFO", "DSN_DOWNLINK", f"Flushed {synced_count} buffered images to Earth")
        return synced_count


# ==========================================
# 3. 60M AI LOOKAHEAD SAFETY ENGINE
# ==========================================
class AutonomousSafetyEngine:
    def __init__(self, vault: SouthPoleVault):
        self.vault = vault

    def predict_forward_corridor(
        self, dem: np.ndarray, thermal_k: np.ndarray, rover_pos: tuple[int, int], facing: str, lookahead_steps: int = 6
    ) -> dict:
        rows, cols = dem.shape
        r, c = rover_pos
        gy, gx = np.gradient(dem * 16.0)
        slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

        dir_vectors = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1), "SOUTH-EAST": (1, 1)}
        dr, dc = dir_vectors.get(facing, (1, 1))
        perp_r, perp_c = -dc, dr

        lanes = {}
        for lane_name, lat_off in {"LEFT": -2, "CENTER": 0, "RIGHT": 2}.items():
            max_slope = 0.0
            min_temp = 999.0
            for step in range(1, lookahead_steps + 1):
                sr = int(np.clip(r + dr * step + perp_r * lat_off, 0, rows - 1))
                sc = int((c + dc * step + perp_c * lat_off) % cols)
                max_slope = max(max_slope, float(slope_deg[sr, sc]))
                min_temp = min(min_temp, float(thermal_k[sr, sc]))
            lanes[lane_name] = {
                "max_slope_deg": round(max_slope, 1),
                "min_temp_k": round(min_temp, 1),
                "safe": max_slope <= 20.0,
            }

        if lanes["CENTER"]["safe"]:
            action = "PROCEED CENTER (CLEAR 60m)"
        elif lanes["LEFT"]["safe"]:
            action = "VEER LEFT (CENTER SLOPE HAZARD)"
        elif lanes["RIGHT"]["safe"]:
            action = "VEER RIGHT (CENTER SLOPE HAZARD)"
        else:
            action = "HALT & RE-ROUTE (CRATER RIM AHEAD)"

        return {"recommended_action": action, "lanes": lanes, "hazard_detected": not lanes["CENTER"]["safe"]}


# ==========================================
# 4. AUTO-UNPACKER & AI MODEL TRAINER
# ==========================================
def unpack_and_catalog_zips(vault: SouthPoleVault) -> dict:
    search_roots = [WORKSPACE_DIR, ISRO_DATA_DIR, WORKSPACE_DIR / "datasets"]
    unzipped_count = 0

    for root in search_roots:
        if not root.exists():
            continue
        iterator = root.glob("*.zip") if root == WORKSPACE_DIR else root.rglob("*.zip")
        for zfile in iterator:
            target = UNPACKED_DIR / zfile.stem
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zfile, "r") as zf:
                        zf.extractall(target)
                    unzipped_count += 1
                except Exception:
                    continue

    xml_meta = {}
    for root in [ISRO_DATA_DIR, WORKSPACE_DIR / "datasets"]:
        if not root.exists():
            continue
        for xfile in root.rglob("*.xml"):
            try:
                tree = ET.parse(xfile)
                lid, inst = "N/A", "PDS4-Instrument"
                for elem in tree.iter():
                    tag = elem.tag.split("}")[-1]
                    if tag == "logical_identifier" and elem.text:
                        lid = elem.text.strip()
                    elif tag == "instrument_name" and elem.text:
                        inst = elem.text.strip()
                xml_meta[xfile.stem] = (inst, lid)
            except Exception:
                continue

    cataloged = 0
    valid_exts = {".csv", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit"}
    for root in [ISRO_DATA_DIR, WORKSPACE_DIR / "datasets"]:
        if not root.exists():
            continue
        for fpath in root.rglob("*"):
            if not fpath.is_file() or fpath.suffix.lower() not in valid_exts:
                continue
            plow = str(fpath).lower()
            if "\\raw\\" in plow or "/raw/" in plow or "inventory" in plow:
                continue

            mission = "Chandrayaan-2" if ("chandrayaan_2" in plow or "ch2" in plow) else "Chandrayaan-3"
            inst, lid = xml_meta.get(fpath.stem, (fpath.suffix.upper(), "N/A"))
            if "ch3_cht_cal" in plow or "chaste" in plow:
                inst = "ChaSTE Thermal Probe (Calibrated)"
            elif "ch3_rim_nc1" in plow or "\\rim\\" in plow or "/rim/" in plow:
                inst = "Pragyan Rover Imager (NavCam-1)"
            elif "libs" in plow:
                inst = "LIBS Laser Spectrometer"

            vault.register_dataset_file(mission, fpath, instrument=inst, pds4_lid=lid)
            cataloged += 1

    return {"unzipped_archives": unzipped_count, "cataloged_files": cataloged}


def extract_chaste_kelvin_samples(csv_paths: list[Path], max_files: int = 12) -> np.ndarray:
    temps = []
    cal_files = [p for p in csv_paths if "ch3_cht_cal" in p.name.lower()] or csv_paths
    for cf in cal_files[:max_files]:
        try:
            df = pd.read_csv(cf)
            num_df = df.select_dtypes(include=[np.number])
            for col in num_df.columns:
                vals = num_df[col].dropna().values
                valid_k = vals[(vals >= 40.0) & (vals <= 400.0)]
                if len(valid_k) > 10:
                    temps.extend(valid_k[:: max(1, len(valid_k) // 80)].tolist())
        except Exception:
            continue
    return np.array(temps, dtype=np.float32)


def _load_raster_normalized(file_path: Path, size: int = 48) -> np.ndarray:
    if file_path.suffix.lower() in {".fits", ".fit"} and HAS_ASTROPY:
        with fits.open(file_path) as hdul:
            arr = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2:
                    arr = np.asarray(hdu.data, dtype=np.float32)
                    break
            if arr is None:
                arr = np.zeros((size, size), dtype=np.float32)
            while arr.ndim > 2:
                arr = arr[0]
            arr = np.asarray(Image.fromarray(arr).resize((size, size)), dtype=np.float32)
    else:
        img = Image.open(file_path)
        if img.mode not in ("L", "F", "I;16"):
            img = img.convert("L")
        arr = np.asarray(img.resize((size, size)), dtype=np.float32)

    valid = arr[arr > -9000]
    vmin = float(valid.min()) if valid.size > 0 else float(arr.min())
    vmax = float(valid.max()) if valid.size > 0 else float(arr.max())
    arr = np.clip(arr, vmin, vmax)
    denom = vmax - vmin
    return (arr - vmin) / (denom if denom > 1e-6 else 1.0)


def extract_features_from_dem(dem: np.ndarray, thermal_k: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dem * 16.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    roughness = np.abs(gx) + np.abs(gy)
    return np.column_stack([dem.ravel(), slope_deg.ravel(), thermal_k.ravel(), roughness.ravel()])


def _fit_and_save_model(X: np.ndarray, y: np.ndarray, source: str) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=65, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, clf.predict(X_test)))
    bundle = {"model": clf, "accuracy": round(acc * 100.0, 2), "source": source}
    joblib.dump(bundle, MODEL_PATH)
    return {"accuracy_pct": bundle["accuracy"], "source": source}


def train_default_southpole_model() -> dict:
    rng = np.random.default_rng(2026)
    n = 2500
    elev = rng.uniform(0.0, 1.0, n)
    slope = rng.uniform(0.0, 42.0, n)
    temp = 40.0 + (elev ** 1.4) * 220.0 + rng.normal(0, 6.0, n)
    rough = slope / 24.0 + rng.normal(0, 0.04, n)

    X = np.column_stack([elev, slope, temp, rough])
    y = [
        "CRATER_WALL_HAZARD" if s > 20.0
        else ("CANDIDATE_PSR_COLD_TRAP" if t < 105.0 else ("SUNLIT_PEAK_COMMS" if e > 0.75 else "SAFE_REGOLITH"))
        for e, s, t in zip(elev, slope, temp)
    ]
    return _fit_and_save_model(X, np.array(y), source="CH3_SHIV_SHAKTI_PHYSICS_MODEL")


def train_from_all_archives(vault: SouthPoleVault) -> dict:
    unpack_and_catalog_zips(vault)
    valid_img_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if HAS_ASTROPY:
        valid_img_exts.update({".fits", ".fit"})

    rasters, csv_files = [], []
    seen = set()
    for root in [ISRO_DATA_DIR, WORKSPACE_DIR / "datasets"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()))
                plow = str(p).lower()
                if "\\raw\\" in plow or "/raw/" in plow or "inventory" in plow:
                    continue
                if p.suffix.lower() in valid_img_exts:
                    rasters.append(p)
                elif p.suffix.lower() == ".csv":
                    csv_files.append(p)

    if not rasters and not csv_files:
        return train_default_southpole_model()

    chaste_temps = extract_chaste_kelvin_samples(csv_files)
    t_min = float(np.percentile(chaste_temps, 5)) if chaste_temps.size > 0 else 40.0
    t_max = float(np.percentile(chaste_temps, 95)) if chaste_temps.size > 0 else 280.0

    X_list, y_list = [], []
    if rasters:
        for raster_file in rasters[:35]:
            arr = _load_raster_normalized(raster_file, size=48)
            temp_k = t_min + (arr ** 1.3) * max(60.0, (t_max - t_min))
            feats = extract_features_from_dem(arr, temp_k)
            for elev, slope, temp, rough in feats:
                if slope > 20.0:
                    lbl = "CRATER_WALL_HAZARD"
                elif temp < 105.0:
                    lbl = "CANDIDATE_PSR_COLD_TRAP"
                elif elev > 0.75:
                    lbl = "SUNLIT_PEAK_COMMS"
                else:
                    lbl = "SAFE_REGOLITH"
                X_list.append([elev, slope, temp, rough])
                y_list.append(lbl)
    else:
        rng = np.random.default_rng(2026)
        n = 2500
        elev = rng.uniform(0.0, 1.0, n)
        slope = rng.uniform(0.0, 42.0, n)
        temp = t_min + (elev ** 1.35) * max(60.0, (t_max - t_min)) + rng.normal(0, 4.0, n)
        rough = slope / 24.0 + rng.normal(0, 0.04, n)
        for e, s, t, r in zip(elev, slope, temp, rough):
            lbl = (
                "CRATER_WALL_HAZARD" if s > 20.0
                else ("CANDIDATE_PSR_COLD_TRAP" if t < 105.0 else ("SUNLIT_PEAK_COMMS" if e > 0.75 else "SAFE_REGOLITH"))
            )
            X_list.append([e, s, t, r])
            y_list.append(lbl)

    chaste_tag = f"+ChaSTE({len(chaste_temps)}pts)" if chaste_temps.size > 0 else ""
    img_tag = f"CH3_RIM({len(rasters)}imgs)" if rasters else "CH3_THERMAL_ONLY"
    return _fit_and_save_model(np.array(X_list), np.array(y_list), source=f"{img_tag}{chaste_tag}")


@st.cache_resource
def init_backend_systems():
    vault = SouthPoleVault()
    engine = AutonomousSafetyEngine(vault)
    stats = train_from_all_archives(vault)
    bundle = joblib.load(MODEL_PATH)
    vault.log_event("INFO", "STARTUP", f"AI trained on startup: {stats['accuracy_pct']}% ({stats['source']})")
    return vault, engine, bundle["model"], bundle["accuracy"], bundle["source"]


vault, safety_engine, sp_ai_model, ai_accuracy, ai_source = init_backend_systems()


@st.cache_data
def discover_chandrayaan_assets():
    rim_imgs, chaste_csvs = [], []
    valid_img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for root in [ISRO_DATA_DIR, WORKSPACE_DIR / "datasets"]:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            plow = str(p).lower()
            if "\\raw\\" in plow or "/raw/" in plow or "inventory" in plow:
                continue
            if p.suffix.lower() in valid_img_exts:
                rim_imgs.append(str(p))
            elif p.suffix.lower() == ".csv":
                chaste_csvs.append(p)
    cal_temps = extract_chaste_kelvin_samples(chaste_csvs, max_files=10)
    return rim_imgs, cal_temps


real_rim_images, real_chaste_temps = discover_chandrayaan_assets()


# ==========================================
# 5. SESSION STATE & THEME SETUP
# ==========================================
defaults = {
    "night_mode": True,
    "rover_pos": [68, 24],
    "facing": "SOUTH-EAST",
    "battery": 100.0,
    "distance_m": 0.0,
    "signal_pct": 94.0,
    "sun_elev_deg": 14.0,
    "max_tilt_deg": 25.0,
    "safety_backup_active": False,
    "mission_aborted": False,
    "eco_hibernate": False,
    "status_log": f"AI TRAINED ({ai_source} • {ai_accuracy}%) — SHIV SHAKTI POINT",
    "breadcrumbs": [(time.time(), 68, 24)],
    "sector_seed": 42,
    "planet_model": "Model 1: 3D Spherical Moon",
    "morph_pct": 0,
    "cam_mode": "Lunar Surface Outlay",
    "cam_preset": "Track Rover South Pole",
    "az_deg": 45,
    "el_deg": -25,
    "dem_bytes": None,
    "use_webcam_cam1": False,
    "rim_frame_idx": 0,
    "last_captured_photo": None,
    "processed_zips": set(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "scanned_mask" not in st.session_state:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    rr, cc = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    mask[(rr - 68) ** 2 + (cc - 24) ** 2 <= 14 ** 2] = True
    st.session_state.scanned_mask = mask

is_night = st.session_state.night_mode
if is_night:
    bg_color, card_bg, text_color, sub_text, accent_color, border_color, plotly_tpl = (
        "#05070D", "#0F1623", "#F3F4F6", "#9CA3AF", "#00FFAA", "#1E293B", "plotly_dark"
    )
else:
    bg_color, card_bg, text_color, sub_text, accent_color, border_color, plotly_tpl = (
        "#F1F5F9", "#FFFFFF", "#0F172A", "#475569", "#0284C7", "#CBD5E1", "plotly_white"
    )

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stApp {{background-color: {bg_color}; color: {text_color};}}
    .block-container {{padding-top: 0.7rem; padding-bottom: 1.0rem; max-width: 1540px;}}
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 8px 12px;
        border-radius: 10px;
    }}
    div[data-testid="stMetric"] label {{color: {sub_text} !important;}}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{color: {text_color} !important; font-size: 1.08rem !important;}}
    .cmd-panel {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-top: 3px solid {accent_color};
        border-radius: 12px;
        padding: 10px 12px;
    }}
    .dpad-dock {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 6. LUNAR GEOMETRY, PHYSICS & NAVIGATION
# ==========================================
@st.cache_data
def generate_lunar_globe(seed: int = 42, size: int = 96):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-np.pi / 2, np.pi / 2, size)
    lon = np.linspace(0, 2 * np.pi, size, endpoint=False)
    llon, llat = np.meshgrid(lon, lat)

    dem = (
        0.48
        + 0.16 * np.cos(2 * llon + 0.4) * np.cos(2 * llat)
        + 0.09 * np.sin(4 * llon - 2 * llat)
        + 0.04 * np.cos(9 * llon) * np.sin(7 * llat)
        + rng.normal(0, 0.012, (size, size))
    )

    lunar_craters = [
        (-1.21, 0.56, 0.21, 0.45, "Shiv Shakti Point (CH-3)"),
        (-1.38, 1.55, 0.24, 0.55, "Shackleton Crater (PSR)"),
        (-1.22, 2.35, 0.28, 0.62, "Cabeus Cold Trap"),
        (-1.12, 0.85, 0.22, 0.48, "Malapert Massif"),
        (-1.30, 3.60, 0.25, 0.52, "Shoemaker Crater"),
        (-0.85, 1.90, 0.30, 0.58, "Tycho Ray Basin"),
        (0.18, 1.45, 0.34, 0.65, "Copernicus Crater"),
        (-0.25, 4.45, 0.32, 0.60, "Mare Tranquillitatis"),
    ]

    for _ in range(28):
        m_lat = float(rng.uniform(-1.45, 1.45))
        m_lon = float(rng.uniform(0, 2 * np.pi))
        m_rad = float(rng.uniform(0.06, 0.13))
        m_dep = float(rng.uniform(0.18, 0.35))
        cos_d = np.sin(llat) * np.sin(m_lat) + np.cos(llat) * np.cos(m_lat) * np.cos(llon - m_lon)
        ang_d = np.arccos(np.clip(cos_d, -1.0, 1.0))
        dem += -m_dep * np.exp(-(ang_d / (m_rad * 0.6)) ** 2) + (m_dep * 0.32) * np.exp(-((ang_d - m_rad) / (m_rad * 0.25)) ** 2)

    crater_markers = []
    for clat, clon, rad, depth, label in lunar_craters:
        cos_d = np.sin(llat) * np.sin(clat) + np.cos(llat) * np.cos(clat) * np.cos(llon - clon)
        ang_dist = np.arccos(np.clip(cos_d, -1.0, 1.0))
        dem += -depth * np.exp(-(ang_dist / (rad * 0.60)) ** 2) + (depth * 0.38) * np.exp(-((ang_dist - rad) / (rad * 0.22)) ** 2)
        gr = int(((np.pi / 2 - clat) / np.pi) * (size - 1))
        gc = int((clon / (2 * np.pi)) * (size - 1))
        crater_markers.append((gr, gc, label))

    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    return dem, crater_markers


def bytes_to_lunar_dem(img_bytes: bytes, size: int = 96):
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flat_idx = np.argsort(arr.ravel())[:4]
    markers = [(int(i // size), int(i % size), f"Candidate PSR-{idx+1}") for idx, i in enumerate(flat_idx)]
    return arr, markers


def compute_lunar_physics(dem: np.ndarray, sun_elev_deg: float):
    gy, gx = np.gradient(dem * 16.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    lat_rad = np.linspace(np.pi / 2, -np.pi / 2, dem.shape[0])[:, None]
    solar_factor = np.clip(np.cos(lat_rad) + math.sin(math.radians(sun_elev_deg)), 0.05, 1.2)

    if real_chaste_temps.size > 0:
        t_low = float(np.percentile(real_chaste_temps, 5))
        t_high = float(np.percentile(real_chaste_temps, 95))
        thermal_k = t_low + (dem ** 1.35) * max(50.0, (t_high - t_low)) * solar_factor
    else:
        thermal_k = 40.0 + (dem ** 1.35) * 185.0 * solar_factor + 35.0

    return slope_deg, thermal_k


def grid_to_sphere_xyz(r_vals, c_vals, dem: np.ndarray, r_base: float = 10.0, relief: float = 0.85, offset: float = 0.0):
    r_arr = np.asarray(r_vals, dtype=float)
    c_arr = np.asarray(c_vals, dtype=float)
    theta = (r_arr / (GRID_SIZE - 1)) * np.pi
    phi = (c_arr / GRID_SIZE) * (2.0 * np.pi)
    ri = np.clip(np.round(r_arr).astype(int), 0, GRID_SIZE - 1)
    ci = np.mod(np.round(c_arr).astype(int), GRID_SIZE)
    radius = r_base + dem[ri, ci] * relief + offset
    return radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)


def grid_to_morph_xyz(r_vals, c_vals, dem: np.ndarray, curr_r: int, curr_c: int, morph_t: float, offset: float = 0.0, mask_far: bool = True):
    r_arr = np.asarray(r_vals, dtype=float)
    c_arr = np.asarray(c_vals, dtype=float)
    r_base = 10.0
    relief = 0.85 * (1.0 + 1.8 * morph_t)

    xs, ys, zs = grid_to_sphere_xyz(r_arr, c_arr, dem, r_base=r_base, relief=relief, offset=offset)
    theta_r = (curr_r / (GRID_SIZE - 1)) * np.pi
    phi_r = (curr_c / GRID_SIZE) * (2.0 * np.pi)

    x1 = xs * np.cos(phi_r) + ys * np.sin(phi_r)
    y1 = -xs * np.sin(phi_r) + ys * np.cos(phi_r)
    x_rot = x1 * np.cos(theta_r) - zs * np.sin(theta_r)
    z_rot = x1 * np.sin(theta_r) + zs * np.cos(theta_r)

    dr = r_arr - curr_r
    dc = ((c_arr - curr_c + GRID_SIZE / 2.0) % GRID_SIZE) - GRID_SIZE / 2.0
    ri = np.clip(np.round(r_arr).astype(int), 0, GRID_SIZE - 1)
    ci = np.mod(np.round(c_arr).astype(int), GRID_SIZE)

    xm = (1.0 - morph_t) * x_rot + morph_t * (dr * 0.38)
    ym = (1.0 - morph_t) * y1 + morph_t * (dc * 0.38)
    zm = (1.0 - morph_t) * z_rot + morph_t * (r_base + dem[ri, ci] * relief + offset)

    if mask_far and morph_t > 0.04:
        outside = np.hypot(dr, dc) > ((1.0 - morph_t) * 68.0 + morph_t * 22.0)
        xm = np.where(outside, np.nan, xm)
        ym = np.where(outside, np.nan, ym)
        zm = np.where(outside, np.nan, zm)

    return xm, ym, zm


def analyze_targeted_object(dem: np.ndarray, ai_grid: np.ndarray, slope_deg: np.ndarray, tr: int, tc: int):
    raw_class = str(ai_grid[tr, tc])
    local_slope = float(slope_deg[tr, tc])
    r0, r1 = max(0, tr - 3), min(GRID_SIZE, tr + 4)
    c0, c1 = max(0, tc - 3), min(GRID_SIZE, tc + 4)
    patch = dem[r0:r1, c0:c1] * 14.0
    height_diff = float(patch.max() - patch.min())
    dim_str = (
        f"{round((c1 - c0) * CELL_METERS + height_diff * 1.3, 1)}m × "
        f"{round((r1 - r0) * CELL_METERS + height_diff * 1.1, 1)}m × "
        f"{round(max(0.5, height_diff), 1)}m (H)"
    )

    if local_slope > st.session_state.max_tilt_deg:
        return f"TIP HAZARD (>{st.session_state.max_tilt_deg:.0f}°)", dim_str

    friendly = {
        "CRATER_WALL_HAZARD": "Lunar Crater Rim Wall",
        "CANDIDATE_PSR_COLD_TRAP": "Candidate PSR Cold Trap",
        "SUNLIT_PEAK_COMMS": "Sunlit Peak (Solar Surf)",
        "SAFE_REGOLITH": "Lunar Mare Regolith",
    }
    return friendly.get(raw_class, raw_class), dim_str


def get_10min_retrace_path():
    cutoff = time.time() - 600.0
    valid = [(r, c) for (ts, r, c) in st.session_state.breadcrumbs if ts >= cutoff]
    if not valid:
        return [tuple(st.session_state.rover_pos)]
    dedup = []
    for pt in valid:
        if not dedup or dedup[-1] != pt:
            dedup.append(pt)
    return dedup[::-1]


def get_next_scan_target(ai_grid: np.ndarray, slope_deg: np.ndarray, scanned_mask: np.ndarray, curr: tuple):
    max_t = st.session_state.max_tilt_deg
    unmapped = np.argwhere(~scanned_mask & (slope_deg <= max_t))
    if len(unmapped) == 0:
        return curr
    dists = np.hypot(unmapped[:, 0] - curr[0], unmapped[:, 1] - curr[1])
    best_target = tuple(unmapped[int(np.argmin(dists))])

    rows, cols = ai_grid.shape
    open_set = [(0.0, curr)]
    came_from = {}
    g_score = {curr: 0.0}
    neighbors = [(-2, 0, 2.0), (2, 0, 2.0), (0, -2, 2.0), (0, 2, 2.0), (-2, -2, 2.82), (-2, 2, 2.82), (2, -2, 2.82), (2, 2, 2.82)]

    while open_set:
        _, node = heapq.heappop(open_set)
        if math.hypot(node[0] - best_target[0], node[1] - best_target[1]) <= 3.0:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path = path[::-1]
            return path[1] if len(path) > 1 else curr
        for dr, dc, step_d in neighbors:
            nr, nc = (node[0] + dr), (node[1] + dc) % cols
            if 2 <= nr < rows - 2 and slope_deg[nr, nc] <= max_t:
                tg = g_score[node] + step_d
                if tg < g_score.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = node
                    g_score[(nr, nc)] = tg
                    heapq.heappush(open_set, (tg + math.hypot(best_target[0] - nr, best_target[1] - nc), (nr, nc)))
    return curr


def load_all_db_tables():
    try:
        conn = sqlite3.connect(DB_PATH)
        media_df = pd.read_sql_query(
            "SELECT id, timestamp, image_file, grid_r, grid_c, temp_k, ai_prediction, synced_to_earth FROM southpole_media_vault ORDER BY id DESC LIMIT 15",
            conn,
        )
        crumb_df = pd.read_sql_query(
            "SELECT id, timestamp, grid_r, grid_c, lat, lon, signal_pct, mode FROM breadcrumb_trail ORDER BY id DESC LIMIT 15",
            conn,
        )
        ds_df = pd.read_sql_query(
            "SELECT id, mission_name, filename, file_format, instrument_name, pds4_lid FROM uploaded_datasets ORDER BY id DESC LIMIT 30",
            conn,
        )
        logs_df = pd.read_sql_query(
            "SELECT id, timestamp, log_level, module, message FROM mission_logs ORDER BY id DESC LIMIT 15",
            conn,
        )
        conn.close()
        return media_df, crumb_df, ds_df, logs_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def synthesize_lunar_horizon(dem: np.ndarray, r: int, c: int, facing: str):
    patch = np.roll(np.roll(dem, -r + 20, axis=0), -c + 20, axis=1)[:40, :40]
    rot_map = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3, "SOUTH-EAST": 2}
    patch = np.rot90(patch, rot_map.get(facing, 0))

    canvas = np.zeros((225, 460, 3), dtype=np.uint8)
    rng = np.random.default_rng(r * 131 + c)
    for _ in range(40):
        canvas[int(rng.integers(4, 80)), int(rng.integers(4, 456))] = [235, 240, 250]

    for row_idx in range(40):
        elev_line = np.interp(np.arange(460), np.linspace(0, 459, 40), patch[row_idx])
        y_base = int(82 + (row_idx / 40.0) * 140)
        shade = int(38 + (row_idx / 40.0) * 185)
        for x in range(460):
            y_top = max(58, int(y_base - elev_line[x] * 58))
            canvas[y_top:y_base, x] = [shade, shade, int(min(255, shade * 1.03))]

    return Image.fromarray(canvas)


def get_cam1_source_frame(dem: np.ndarray, r: int, c: int, facing: str):
    if real_rim_images:
        idx = st.session_state.rim_frame_idx % len(real_rim_images)
        return Image.open(Path(real_rim_images[idx])), f"CH3-RIM [{idx+1}/{len(real_rim_images)}]"
    return synthesize_lunar_horizon(dem, r, c, facing), "PRAGYAN-SIM"


def render_cam_hud(
    pil_img: Image.Image, cam_title: str, thermal: bool, target_pos_str: str, obj_id: str, dim_str: str, temp_k: float, alert_mode: str | None
):
    img = pil_img.resize((460, 225))
    img = ImageOps.colorize(ImageOps.grayscale(img), black="#0D0887", mid="#CC4778", white="#F0F921") if thermal else img.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    hud_color = "#EF4444" if alert_mode else ("#FACC15" if thermal else "#00FFAA")

    draw.rectangle([w // 2 - 68, h // 2 - 36, w // 2 + 68, h // 2 + 36], outline=hud_color, width=2)
    draw.line([(w // 2 - 12, h // 2), (w // 2 + 12, h // 2)], fill=hud_color, width=1)
    draw.line([(w // 2, h // 2 - 12), (w // 2, h // 2 + 12)], fill=hud_color, width=1)

    draw.rectangle([0, 0, w, 23], fill=(8, 12, 20))
    draw.text((10, 5), f"{cam_title} | {alert_mode if alert_mode else target_pos_str}", fill=hud_color)

    draw.rectangle([0, h - 25, w, h], fill=(8, 12, 20))
    draw.text((10, h - 18), f"ID: {obj_id} | DIM: {dim_str} | {temp_k:.0f}K", fill="#FFFFFF")
    return img


def move_rover(new_r: int, new_c: int, facing_dir: str, fuel_cost: float, scan_rad: int = 13, is_retrace: bool = False):
    if st.session_state.mission_aborted:
        return
    eff_cost = fuel_cost * (0.45 if st.session_state.eco_hibernate else 1.0)
    old_r, old_c = st.session_state.rover_pos
    new_r = int(np.clip(new_r, 2, GRID_SIZE - 3))
    new_c = int(new_c % GRID_SIZE)

    step_cells = math.hypot(new_r - old_r, min(abs(new_c - old_c), GRID_SIZE - abs(new_c - old_c)))
    st.session_state.distance_m += step_cells * CELL_METERS
    st.session_state.rover_pos = [new_r, new_c]
    st.session_state.facing = facing_dir
    st.session_state.battery = max(0.0, st.session_state.battery - eff_cost)

    if real_rim_images:
        st.session_state.rim_frame_idx = (st.session_state.rim_frame_idx + 1) % len(real_rim_images)

    rr, cc = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    dc = np.minimum(np.abs(cc - new_c), GRID_SIZE - np.abs(cc - new_c))
    st.session_state.scanned_mask[(rr - new_r) ** 2 + dc ** 2 <= scan_rad ** 2] = True

    lat_now = round(90.0 - (new_r / (GRID_SIZE - 1)) * 180.0, 2)
    lon_now = round((new_c / GRID_SIZE) * 360.0 - 180.0, 2)

    if not is_retrace:
        st.session_state.breadcrumbs.append((time.time(), new_r, new_c))
        vault.record_breadcrumb(new_r, new_c, lat_now, lon_now, st.session_state.signal_pct, "EXPLORE")
    else:
        vault.record_breadcrumb(new_r, new_c, lat_now, lon_now, st.session_state.signal_pct, "SAFETY_BACKUP")


def execute_retrace_step():
    st.session_state.safety_backup_active = True
    retrace_pts = get_10min_retrace_path()
    if len(retrace_pts) > 1:
        next_back_r, next_back_c = retrace_pts[1]
        if len(st.session_state.breadcrumbs) > 1:
            st.session_state.breadcrumbs.pop()
        move_rover(next_back_r, next_back_c, st.session_state.facing, 0.4, is_retrace=True)
        st.session_state.status_log = f"SAFETY BACKUP RETRACING -> ({next_back_r}, {next_back_c})"
        vault.log_event("WARN", "SAFETY_BACKUP", f"Retraced step to ({next_back_r}, {next_back_c})")
        if len(st.session_state.breadcrumbs) <= 1:
            st.session_state.signal_pct = 90.0
            st.session_state.safety_backup_active = False
            flushed = vault.flush_downlink_queue()
            st.session_state.status_log = f"RETRACE COMPLETE — SIGNAL RESTORED & {flushed} IMAGES SYNCED"
    else:
        st.session_state.status_log = "ROVER AT 10-MIN ORIGIN BEACON — HOLD POSITION"


# ==========================================
# 7. TOP BAR: TITLE + ZIP UPLOADER + THEME + MENU
# ==========================================
top_title, top_zip, top_theme, top_menu = st.columns([4.8, 3.2, 1.2, 0.6], vertical_alignment="center")
with top_title:
    st.markdown("### ASTRA-SCAN &nbsp;|&nbsp; Chandrayaan Rover OS")

with top_zip:
    uploaded_zips = st.file_uploader(
        "Drop Additional ISRO .ZIP Archive",
        type=["zip"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_zips:
        new_added = False
        for zf in uploaded_zips:
            sig = (zf.name, zf.size)
            if sig not in st.session_state.processed_zips:
                dest_zip = ISRO_DATA_DIR / zf.name
                dest_zip.write_bytes(zf.getvalue())
                st.session_state.processed_zips.add(sig)
                new_added = True
        if new_added:
            with st.spinner("Unpacking archive & retraining AI..."):
                stats = train_from_all_archives(vault)
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state.status_log = f"ZIP TRAINED AI: {stats['accuracy_pct']}% ({stats['source']})"
                vault.log_event("INFO", "ZIP_INGEST", st.session_state.status_log)
            st.rerun()

with top_theme:
    if st.button("☀️ Day Mode" if st.session_state.night_mode else "🌙 Night Mode", use_container_width=True):
        st.session_state.night_mode = not st.session_state.night_mode
        st.rerun()

with top_menu:
    with st.popover("⋮"):
        st.markdown("#### Mission & What-If Simulator")
        st.session_state.cam_mode = st.radio(
            "Scientific Sensor Layer",
            ["Lunar Surface Outlay", "ChaSTE Cryogenic Thermal (K)", "AI South Pole Geology"],
            index=["Lunar Surface Outlay", "ChaSTE Cryogenic Thermal (K)", "AI South Pole Geology"].index(st.session_state.cam_mode),
        )
        st.session_state.cam_preset = st.selectbox(
            "3D Moon Camera Angle",
            ["Track Rover South Pole", "South Pole Overhead", "Full Lunar Globe", "Custom Angle"],
        )
        if st.session_state.cam_preset == "Custom Angle":
            st.session_state.az_deg = st.slider("Azimuth (°)", 0, 360, st.session_state.az_deg)
            st.session_state.el_deg = st.slider("Elevation (°)", -85, 85, st.session_state.el_deg)

        st.divider()
        new_sun = st.slider("Sun Elevation Angle (°)", 2.0, 60.0, float(st.session_state.sun_elev_deg))
        new_tilt = st.slider("Pragyan Max Tilt Limit (°)", 10.0, 35.0, float(st.session_state.max_tilt_deg))
        if new_sun != st.session_state.sun_elev_deg or new_tilt != st.session_state.max_tilt_deg:
            st.session_state.sun_elev_deg = new_sun
            st.session_state.max_tilt_deg = new_tilt
            vault.log_simulation(new_sun, new_tilt, st.session_state.battery, 0.0, f"What-If: Sun={new_sun}°, Tilt={new_tilt}°")
            st.rerun()

        st.divider()
        up_file = st.file_uploader("Upload Single Raster (TIF/JPG/FITS)", type=["tif", "tiff", "png", "jpg", "jpeg", "fits"])
        if up_file is not None:
            st.session_state.dem_bytes = up_file.getvalue()
        st.session_state.sector_seed = st.slider("Lunar Terrain Seed", 1, 30, st.session_state.sector_seed)
        st.caption(f"AI Model: {ai_source} ({ai_accuracy}%)")


# ==========================================
# 8. COMPUTE TERRAIN, AI PREDICTIONS & TELEMETRY
# ==========================================
if st.session_state.dem_bytes is not None:
    dem, crater_landmarks = bytes_to_lunar_dem(st.session_state.dem_bytes, GRID_SIZE)
else:
    dem, crater_landmarks = generate_lunar_globe(st.session_state.sector_seed, GRID_SIZE)

slope_deg, thermal_k = compute_lunar_physics(dem, st.session_state.sun_elev_deg)
feats = extract_features_from_dem(dem, thermal_k)
ai_outlay_grid = sp_ai_model.predict(feats).reshape(GRID_SIZE, GRID_SIZE)

curr_r, curr_c = st.session_state.rover_pos
facing_offsets = {"NORTH": (-5, 0), "SOUTH": (5, 0), "EAST": (0, 5), "WEST": (0, -5), "SOUTH-EAST": (4, 4)}
facing_azimuth = {"NORTH": 0, "EAST": 90, "SOUTH": 180, "WEST": 270, "SOUTH-EAST": 135}
off_r, off_c = facing_offsets.get(st.session_state.facing, (4, 4))
targ_r = int(np.clip(curr_r + off_r, 0, GRID_SIZE - 1))
targ_c = int((curr_c + off_c) % GRID_SIZE)

t_lat = round(90.0 - (targ_r / (GRID_SIZE - 1)) * 180.0, 1)
t_lon = round((targ_c / GRID_SIZE) * 360.0 - 180.0, 1)
target_pos_str = f"{t_lat}°, {t_lon}° (Az {facing_azimuth.get(st.session_state.facing, 135)}°)"
obj_identified, target_dimensions = analyze_targeted_object(dem, ai_outlay_grid, slope_deg, targ_r, targ_c)
curr_temp_k = float(thermal_k[targ_r, targ_c])
curr_temp_c = curr_temp_k - 273.15
mapped_pct = float(np.mean(st.session_state.scanned_mask) * 100.0)

local_elev_norm = float(dem[curr_r, curr_c])
solar_gen_w = round(SOLAR_PANEL_MAX_W * local_elev_norm * max(0.08, math.sin(math.radians(st.session_state.sun_elev_deg)) * 2.2), 1)
propulsion_w = round(14.5 + (PRAGYAN_MASS_KG * 0.12 * math.sin(math.radians(slope_deg[curr_r, curr_c]))), 1)
avionics_w = 8.0 if st.session_state.eco_hibernate else 15.0
net_power_w = round(solar_gen_w - (propulsion_w + avionics_w), 1)

lookahead_info = safety_engine.predict_forward_corridor(dem, thermal_k, (curr_r, curr_c), st.session_state.facing, lookahead_steps=6)
retrace_coords = get_10min_retrace_path()
raw_pil_1, cam1_source_badge = get_cam1_source_frame(dem, curr_r, curr_c, st.session_state.facing)
alert_tag = "MISSION ABORTED" if st.session_state.mission_aborted else ("SAFETY BACKUP" if st.session_state.safety_backup_active else None)

st.caption(
    f"{st.session_state.status_log} • Mapped: {mapped_pct:.1f}% • "
    f"Solar Surfing Net: {net_power_w:+.1f} W • 60m Lookahead: {lookahead_info['recommended_action']}"
)

if st.session_state.mission_aborted:
    st.error("🛑 **MISSION ABORT ENGAGED — PROPULSION LOCKED!** Click **✅ Clear Abort & Resume** in the Command Deck.")
elif st.session_state.safety_backup_active:
    st.warning(f"⚠️ **SAFETY BACKUP ACTIVE** — DSN Signal at **{st.session_state.signal_pct:.0f}%**! Retracing 10-min breadcrumb trail ({len(retrace_coords)} pts).")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("ChaSTE Temp", f"{curr_temp_k:.1f} K", f"{curr_temp_c:.1f} °C")
m2.metric("Battery & Solar", f"{st.session_state.battery:.1f}%", f"{net_power_w:+.1f} W Net")
m3.metric("Distance Travelled", f"{st.session_state.distance_m:.1f} m", f"DSN Sig {st.session_state.signal_pct:.0f}%")
m4.metric("Cam Targeted Pos", f"{t_lat}°, {t_lon}°", f"Az {facing_azimuth.get(st.session_state.facing, 135)}°")
m5.metric("Object Identified", obj_identified, f"Slope {slope_deg[targ_r, targ_c]:.1f}°")
m6.metric("Target Dimension", target_dimensions, f"Max Tilt {st.session_state.max_tilt_deg:.0f}°")


# ==========================================
# 9. MAIN 3-COLUMN WORKSPACE
# ==========================================
col_planet, col_cmds, col_cams = st.columns([1.45, 0.72, 1.18])
LUNAR_COLORSCALE = [[0.0, "#12151C"], [0.25, "#2E3440"], [0.50, "#5C6472"], [0.75, "#9BA4B2"], [1.0, "#ECEFF4"]]

with col_planet:
    if st.session_state.cam_mode == "ChaSTE Cryogenic Thermal (K)":
        surf_color, cscale, cbar_lbl = thermal_k, "Inferno", "Temp (K)"
    elif st.session_state.cam_mode == "AI South Pole Geology":
        cmap = {"SAFE_REGOLITH": 0, "CANDIDATE_PSR_COLD_TRAP": 1, "SUNLIT_PEAK_COMMS": 2, "CRATER_WALL_HAZARD": 3}
        surf_color = np.vectorize(lambda k: cmap.get(str(k), 0))(ai_outlay_grid)
        cscale = [[0.0, "#10B981"], [0.33, "#38BDF8"], [0.66, "#FACC15"], [1.0, "#EF4444"]]
        cbar_lbl = "AI Class"
    else:
        surf_color, cscale, cbar_lbl = dem * 14.0, LUNAR_COLORSCALE, "Elev (km)"

    fig_planet = go.Figure()
    active_model = st.session_state.planet_model

    if active_model == "Model 1: 3D Spherical Moon":
        cc_grid, rr_grid = np.meshgrid(np.arange(GRID_SIZE + 1), np.arange(GRID_SIZE))
        X_sph, Y_sph, Z_sph = grid_to_sphere_xyz(rr_grid, cc_grid, dem)
        fig_planet.add_trace(
            go.Surface(
                x=X_sph, y=Y_sph, z=Z_sph, surfacecolor=np.hstack([surf_color, surf_color[:, :1]]), colorscale=cscale,
                colorbar=dict(title=cbar_lbl, len=0.60),
                lighting=dict(ambient=0.58, diffuse=0.90, roughness=0.88, specular=0.08),
            )
        )
        mx, my, mz = grid_to_sphere_xyz([m[0] for m in crater_landmarks], [m[1] for m in crater_landmarks], dem, offset=0.45)
        fig_planet.add_trace(
            go.Scatter3d(
                x=mx, y=my, z=mz, mode="text", text=[m[2] for m in crater_landmarks],
                textfont=dict(color="#FACC15" if is_night else "#B45309", size=10),
            )
        )

        angles = np.linspace(0, 2 * np.pi, 60)
        rx, ry, rz = grid_to_sphere_xyz(
            np.clip(curr_r + 12.0 * np.sin(angles), 0, GRID_SIZE - 1), (curr_c + 12.0 * np.cos(angles)) % GRID_SIZE, dem, offset=0.18
        )
        fig_planet.add_trace(
            go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(color="#EF4444" if st.session_state.mission_aborted else "#00FFAA", width=5))
        )

        if len(retrace_coords) > 1:
            bx, by, bz = grid_to_sphere_xyz([p[0] for p in retrace_coords], [p[1] for p in retrace_coords], dem, offset=0.28)
            retrace_color = "#EF4444" if st.session_state.safety_backup_active else "#F59E0B"
            fig_planet.add_trace(
                go.Scatter3d(
                    x=bx, y=by, z=bz, mode="lines+markers+text",
                    line=dict(color=retrace_color, width=7 if st.session_state.safety_backup_active else 4, dash="dash"),
                    marker=dict(size=4, color=retrace_color),
                    text=["SAFETY BACKUP" if i == 0 else ("ORIGIN" if i == len(retrace_coords) - 1 else "") for i in range(len(retrace_coords))],
                    textposition="top center", textfont=dict(color=retrace_color, size=10),
                )
            )

        px, py, pz = grid_to_sphere_xyz([curr_r, targ_r], [curr_c, targ_c], dem, offset=0.38)
        fig_planet.add_trace(
            go.Scatter3d(
                x=px, y=py, z=pz, mode="markers+text", text=["PRAGYAN ROVER", "TARGET"], textposition="top center",
                marker=dict(size=[8, 7], color=["#00FFAA", "#EF4444"], symbol=["diamond", "cross"]),
            )
        )

        if st.session_state.cam_preset == "Track Rover South Pole":
            r_theta, r_phi = (curr_r / (GRID_SIZE - 1)) * np.pi, (curr_c / GRID_SIZE) * (2.0 * np.pi)
            eye = (1.78 * math.sin(r_theta) * math.cos(r_phi), 1.78 * math.sin(r_theta) * math.sin(r_phi), 1.78 * math.cos(r_theta) - 0.15)
        elif st.session_state.cam_preset == "South Pole Overhead":
            eye = (0.20, 0.20, -1.95)
        elif st.session_state.cam_preset == "Full Lunar Globe":
            eye = (1.60, 1.30, -0.35)
        else:
            az, el = math.radians(st.session_state.az_deg), math.radians(st.session_state.el_deg)
            eye = (1.9 * math.cos(el) * math.cos(az), 1.9 * math.cos(el) * math.sin(az), 1.9 * math.sin(el))

        fig_planet.update_layout(
            title=f"Model 1: 3D Spherical Moon — {st.session_state.cam_mode}",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])), aspectmode="data",
            ),
            height=425, margin=dict(l=0, r=0, t=34, b=0), template=plotly_tpl,
        )

    elif active_model == "Model 2: Small-Distance Lunar Relief":
        half_w = 22
        r_span = np.clip(np.arange(curr_r - half_w, curr_r + half_w + 1), 0, GRID_SIZE - 1)
        c_span = (np.arange(curr_c - half_w, curr_c + half_w + 1)) % GRID_SIZE
        axis_m = np.arange(-half_w, half_w + 1) * CELL_METERS
        fig_planet.add_trace(
            go.Surface(
                x=axis_m, y=axis_m, z=dem[np.ix_(r_span, c_span)] * 14.0, surfacecolor=surf_color[np.ix_(r_span, c_span)], colorscale=cscale,
                colorbar=dict(title=cbar_lbl, len=0.60),
                contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#00FFAA", project_z=True)),
            )
        )
        fig_planet.add_trace(
            go.Scatter3d(
                x=[0.0, off_c * CELL_METERS], y=[0.0, off_r * CELL_METERS],
                z=[dem[curr_r, curr_c] * 14.0 + 0.85, dem[targ_r, targ_c] * 14.0 + 0.85],
                mode="markers+text", text=["PRAGYAN", "TARGET"], textposition="top center",
                marker=dict(size=[8, 7], color=["#00FFAA", "#EF4444"], symbol=["diamond", "cross"]),
            )
        )
        fig_planet.update_layout(
            title="Model 2: Small-Distance Lunar Relief (440m × 440m)",
            scene=dict(
                xaxis_title="East-West (m)", yaxis_title="North-South (m)", zaxis_title="Elevation (m)",
                camera=dict(eye=dict(x=1.35, y=1.35, z=1.15)), aspectmode="manual", aspectratio=dict(x=1.25, y=1.25, z=0.45),
            ),
            height=425, margin=dict(l=0, r=0, t=34, b=0), template=plotly_tpl,
        )

    else:
        morph_t = st.session_state.morph_pct / 100.0
        cc_grid, rr_grid = np.meshgrid(np.arange(GRID_SIZE + 1), np.arange(GRID_SIZE))
        Xm, Ym, Zm = grid_to_morph_xyz(rr_grid, cc_grid, dem, curr_r, curr_c, morph_t, offset=0.0, mask_far=True)
        fig_planet.add_trace(go.Surface(x=Xm, y=Ym, z=Zm, surfacecolor=np.hstack([surf_color, surf_color[:, :1]]), colorscale=cscale, colorbar=dict(title=cbar_lbl, len=0.60)))
        px, py, pz = grid_to_morph_xyz([curr_r, targ_r], [curr_c, targ_c], dem, curr_r, curr_c, morph_t, offset=0.45, mask_far=False)
        fig_planet.add_trace(
            go.Scatter3d(
                x=px, y=py, z=pz, mode="markers+text", text=["PRAGYAN", "TARGET"], textposition="top center",
                marker=dict(size=[8, 7], color=["#00FFAA", "#EF4444"], symbol=["diamond", "cross"]),
            )
        )
        eye = (0.35 + 1.05 * morph_t, 0.35 + 1.05 * morph_t, 1.85 - 0.55 * morph_t)
        fig_planet.update_layout(
            title=f"Model 3: Lunar Sphere → Local Morph ({st.session_state.morph_pct}%)",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])), aspectmode="data",
            ),
            height=425, margin=dict(l=0, r=0, t=34, b=0), template=plotly_tpl,
        )

    st.plotly_chart(fig_planet, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        if st.button("🌑 1: Sphere", use_container_width=True, type="primary" if active_model == "Model 1: 3D Spherical Moon" else "secondary"):
            st.session_state.planet_model = "Model 1: 3D Spherical Moon"
            st.rerun()
    with mc2:
        if st.button("🗺️ 2: Local", use_container_width=True, type="primary" if active_model == "Model 2: Small-Distance Lunar Relief" else "secondary"):
            st.session_state.planet_model = "Model 2: Small-Distance Lunar Relief"
            st.rerun()
    with mc3:
        if st.button("🔄 3: Morph", use_container_width=True, type="primary" if active_model == "Model 3: Spherical ↔ Local Morph" else "secondary"):
            st.session_state.planet_model = "Model 3: Spherical ↔ Local Morph"
            st.rerun()

    if active_model == "Model 3: Spherical ↔ Local Morph":
        new_morph = st.slider("Morph", 0, 100, st.session_state.morph_pct, label_visibility="collapsed")
        if new_morph != st.session_state.morph_pct:
            st.session_state.morph_pct = new_morph
            st.rerun()

with col_cmds:
    st.markdown('<div class="cmd-panel">', unsafe_allow_html=True)
    st.markdown("**🎛️ ROVER COMMAND DECK**")
    st.caption("Mission & Safety Controls")

    if st.button("📸 Take Photo", type="primary", use_container_width=True):
        snap_hud = render_cam_hud(raw_pil_1, f"CAM 1 [{cam1_source_badge}]", False, target_pos_str, obj_identified, target_dimensions, curr_temp_k, alert_tag)
        saved_path = vault.save_southpole_image(snap_hud, curr_r, curr_c, curr_temp_k, obj_identified, signal_ok=(st.session_state.signal_pct >= 25.0))
        st.session_state.last_captured_photo = saved_path
        st.session_state.status_log = f"PHOTO SAVED TO VAULT: {Path(saved_path).name}"
        st.rerun()

    if st.button("⏪ Retrace (Backup)", use_container_width=True, disabled=st.session_state.mission_aborted):
        execute_retrace_step()
        st.rerun()

    if st.button("🤖 AI 60m Scan Step", use_container_width=True, disabled=st.session_state.mission_aborted):
        nr, nc = get_next_scan_target(ai_outlay_grid, slope_deg, st.session_state.scanned_mask, (curr_r, curr_c))
        move_rover(nr, nc, st.session_state.facing, 0.8, scan_rad=14)
        st.session_state.status_log = f"AI LOOKAHEAD STEP -> ({nr}, {nc}) [{lookahead_info['recommended_action']}]"
        vault.log_event("INFO", "AI_NAV", f"Autonomous lookahead step to ({nr}, {nc})")
        st.rerun()

    if st.session_state.safety_backup_active:
        if st.button("🟢 Restore DSN Link", use_container_width=True):
            st.session_state.signal_pct = 92.0
            st.session_state.safety_backup_active = False
            flushed = vault.flush_downlink_queue()
            st.session_state.status_log = f"EARTH LINK RESTORED — {flushed} VAULT PHOTOS SYNCED"
            st.rerun()
    else:
        if st.button("📡 Simulate Comms Loss", use_container_width=True, disabled=st.session_state.mission_aborted):
            st.session_state.signal_pct = 6.0
            st.session_state.safety_backup_active = True
            st.session_state.status_log = "SIGNAL LOST (<25%) — SAFETY BACKUP ENGAGED"
            vault.log_event("WARN", "DSN_COMMS", "Earth DSN signal dropped below 25%; SAFETY BACKUP armed")
            st.rerun()

    if st.button("📦 Re-Train AI on Dataset", use_container_width=True):
        stats = train_from_all_archives(vault)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.status_log = f"TRAINED AI: {stats['accuracy_pct']}% ({stats['source']})"
        vault.log_event("INFO", "ML_TRAINER", st.session_state.status_log)
        st.rerun()

    if st.button("⚡ Exit Eco Hibernate" if st.session_state.eco_hibernate else "🔋 Eco Power Mode", use_container_width=True):
        st.session_state.eco_hibernate = not st.session_state.eco_hibernate
        st.session_state.status_log = "ECO HIBERNATE ON — AVIONICS REDUCED TO 8W" if st.session_state.eco_hibernate else "FULL PROPULSION POWER RESTORED"
        st.rerun()

    st.divider()

    if not st.session_state.mission_aborted:
        if st.button("🛑 MISSION ABORT", type="primary", use_container_width=True):
            st.session_state.mission_aborted = True
            st.session_state.safety_backup_active = False
            st.session_state.status_log = "🛑 MISSION ABORTED — ALL PROPULSION LOCKED"
            vault.log_event("CRITICAL", "COMMAND", "Operator triggered MISSION ABORT emergency stop")
            st.rerun()
    else:
        if st.button("✅ Clear Abort & Resume", use_container_width=True):
            st.session_state.mission_aborted = False
            st.session_state.status_log = "ABORT CLEARED — ROVER SYSTEMS ONLINE"
            vault.log_event("INFO", "COMMAND", "Mission abort cleared by operator")
            st.rerun()

    if st.button("🔄 Full Mission Reset", use_container_width=True):
        st.session_state.rover_pos = [68, 24]
        st.session_state.battery = 100.0
        st.session_state.distance_m = 0.0
        st.session_state.signal_pct = 94.0
        st.session_state.safety_backup_active = False
        st.session_state.mission_aborted = False
        st.session_state.eco_hibernate = False
        st.session_state.breadcrumbs = [(time.time(), 68, 24)]
        st.session_state.dem_bytes = None
        st.session_state.morph_pct = 0
        st.session_state.status_log = "SYSTEM RESET — READY AT SHIV SHAKTI SOUTH POLE"
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        rr, cc = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        mask[(rr - 68) ** 2 + (cc - 24) ** 2 <= 14 ** 2] = True
        st.session_state.scanned_mask = mask
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with col_cams:
    cam1_head, cam1_tog = st.columns([2.1, 1.1])
    with cam1_head:
        st.markdown("**CAM 1 — Pragyan NavCam (RIM)**")
    with cam1_tog:
        st.session_state.use_webcam_cam1 = st.checkbox("Link Webcam", value=st.session_state.use_webcam_cam1)

    if st.session_state.use_webcam_cam1:
        live_frame = st.camera_input("Capture Cam 1 Feed", label_visibility="collapsed")
        if live_frame is not None:
            cam_bytes = live_frame.getvalue()
            st.session_state.dem_bytes = cam_bytes
            raw_pil_1 = Image.open(io.BytesIO(cam_bytes))
            cam1_source_badge = "LIVE-WEBCAM"

    st.image(render_cam_hud(raw_pil_1, f"CAM 1 [{cam1_source_badge}]", False, target_pos_str, obj_identified, target_dimensions, curr_temp_k, alert_tag), use_container_width=True)

    if real_rim_images and not st.session_state.use_webcam_cam1:
        f_col1, f_col2 = st.columns([1.2, 1.2])
        with f_col1:
            if st.button("⏭️ Next CH-3 Photo", use_container_width=True):
                st.session_state.rim_frame_idx = (st.session_state.rim_frame_idx + 1) % len(real_rim_images)
                st.rerun()
        with f_col2:
            if st.button("🗺️ Project Frame to 3D", use_container_width=True):
                active_jpg = real_rim_images[st.session_state.rim_frame_idx % len(real_rim_images)]
                st.session_state.dem_bytes = Path(active_jpg).read_bytes()
                st.session_state.status_log = f"PROJECTED {Path(active_jpg).name} ONTO 3D MOON"
                st.rerun()

    if st.session_state.last_captured_photo:
        sync_lbl = "Queued in Vault (Signal Lost)" if st.session_state.signal_pct < 25.0 else "Synced to Earth DSN"
        st.caption(f"✅ Saved `{Path(st.session_state.last_captured_photo).name}` • **{sync_lbl}**")

    st.markdown("**CAM 2 — ChaSTE Cryogenic Thermal Cam**")
    raw_pil_2 = synthesize_lunar_horizon(dem, targ_r, targ_c, st.session_state.facing)
    st.image(render_cam_hud(raw_pil_2, "CAM 2 [CHASTE-IR]", True, target_pos_str, obj_identified, target_dimensions, curr_temp_k, alert_tag), use_container_width=True)


# ==========================================
# 10. BOTTOM DOCK: D-PAD + THRUSTER + SQLITE DB
# ==========================================
st.markdown('<div class="dpad-dock">', unsafe_allow_html=True)
dock_dpad, dock_jump, dock_db = st.columns([1.05, 1.05, 1.9], vertical_alignment="top")

with dock_dpad:
    st.caption("MANUAL ROVER D-PAD")
    d1, d2, d3 = st.columns(3)
    with d2:
        if st.button("▲ FWD", use_container_width=True, disabled=st.session_state.mission_aborted):
            move_rover(curr_r - 3, curr_c, "NORTH", 0.6)
            st.rerun()
    d4, d5, d6 = st.columns(3)
    with d4:
        if st.button("◄ LEFT", use_container_width=True, disabled=st.session_state.mission_aborted):
            move_rover(curr_r, curr_c - 3, "WEST", 0.6)
            st.rerun()
    with d5:
        if st.button("▼ BACK", use_container_width=True, disabled=st.session_state.mission_aborted):
            move_rover(curr_r + 3, curr_c, "SOUTH", 0.6)
            st.rerun()
    with d6:
        if st.button("► RIGHT", use_container_width=True, disabled=st.session_state.mission_aborted):
            move_rover(curr_r, curr_c + 3, "EAST", 0.6)
            st.rerun()

with dock_jump:
    st.caption("THRUSTER CRATER JUMP")
    j_dir = st.selectbox("Jump Heading", ["SOUTH-EAST", "NORTH", "SOUTH", "EAST", "WEST"], label_visibility="collapsed", disabled=st.session_state.mission_aborted)
    if st.button("🚀 THRUSTER JUMP (80m HOP)", type="primary", use_container_width=True, disabled=st.session_state.mission_aborted):
        if st.session_state.battery >= 6.0:
            j_deltas = {"SOUTH-EAST": (8, 8), "NORTH": (-8, 0), "SOUTH": (8, 0), "EAST": (0, 8), "WEST": (0, -8)}
            dr, dc = j_deltas[j_dir]
            move_rover(curr_r + dr, curr_c + dc, j_dir, 6.0, scan_rad=16)
            st.rerun()
        else:
            st.error("Battery too low (<6%)!")

with dock_db:
    st.caption("RELATIONAL MISSION DATABASE (`rover_mission.db`)")
    media_df, crumb_df, ds_df, logs_df = load_all_db_tables()
    db_t1, db_t2, db_t3, db_t4 = st.tabs(["📸 South Pole Photos", "🧭 10-Min Breadcrumbs", "🛰️ Chandrayaan Datasets", "📋 Mission Logs"])
    with db_t1:
        st.dataframe(media_df, use_container_width=True, height=115) if not media_df.empty else st.info("Click '📸 Take Photo' to save Moon imagery.")
    with db_t2:
        st.dataframe(crumb_df, use_container_width=True, height=115) if not crumb_df.empty else st.info("Drive the rover to record 10-min SAFETY BACKUP breadcrumbs.")
    with db_t3:
        st.dataframe(ds_df, use_container_width=True, height=115) if not ds_df.empty else st.info("Uploaded datasets appear here automatically.")
    with db_t4:
        st.dataframe(logs_df, use_container_width=True, height=115) if not logs_df.empty else st.info("System audit logs appear here.")

st.markdown("</div>", unsafe_allow_html=True)