import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

VAULT_DIR = Path("lunar_southpole_vault")
DB_PATH = "rover_mission.db"


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

            CREATE TABLE IF NOT EXISTS terrain_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                max_slope_deg REAL,
                avg_slope_deg REAL,
                safe_area_pct REAL,
                warning_area_pct REAL,
                hazard_area_pct REAL
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
            ("CH-2", "Chandrayaan-2 Orbiter", "ISRO", "Lunar Polar Orbit", "Active", "TMC-2 DEM & CLASS X-Ray FITS Spectra"),
            ("CH-3", "Chandrayaan-3 Vikram & Pragyan", "ISRO", "Shiv Shakti Point (69.373° S, 32.319° E)", "Completed", "ChaSTE Thermal, LIBS & NavCam Stereo"),
            ("ASTRA", "ASTRA-SCAN South Pole Rover", "ISRO / Astra-2026", "Shackleton-Shiv Shakti Corridor", "Active", "60m AI Lookahead & 10-Min SAFETY BACKUP"),
        ]
        conn.executemany(
            """
            INSERT OR IGNORE INTO missions (code, name, agency, landing_site, status, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
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

    def register_dataset_file(
        self, mission_name: str, file_path: Path, instrument: str = "Unknown", pds4_lid: str = "N/A"
    ):
        conn = sqlite3.connect(self.db_path)
        exists = conn.execute(
            "SELECT id FROM uploaded_datasets WHERE file_path = ?", (str(file_path),)
        ).fetchone()
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

    def get_10min_retrace_waypoints(self, window_sec: float = 600.0, max_steps: int = 30):
        cutoff = time.time() - window_sec
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT id, timestamp, grid_r, grid_c, lat, lon, signal_pct
            FROM breadcrumb_trail
            WHERE epoch_sec >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(cutoff, max_steps),
        )
        conn.close()
        return [(int(row.grid_r), int(row.grid_c)) for _, row in df.iterrows()]

    def save_southpole_image(
        self,
        img_data: np.ndarray | Image.Image,
        r: int,
        c: int,
        temp_k: float,
        ai_pred: str,
        signal_ok: bool,
    ) -> str:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"SP_S{r:02d}_{c:02d}_{ts_tag}.png"
        filepath = self.vault_dir / filename

        if isinstance(img_data, np.ndarray):
            arr = np.clip(img_data * 255.0 if img_data.max() <= 1.0 else img_data, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
        else:
            pil_img = img_data

        pil_img.save(filepath)

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
        cur = conn.execute(
            "UPDATE southpole_media_vault SET synced_to_earth = 1 WHERE synced_to_earth = 0"
        )
        synced_count = cur.rowcount
        conn.commit()
        conn.close()
        if synced_count > 0:
            self.log_event("INFO", "DSN_DOWNLINK", f"Flushed {synced_count} buffered images to Earth")
        return synced_count