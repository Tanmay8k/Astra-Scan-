from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from database_vault import SouthPoleVault

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "southpole_lookahead_ai.pkl"


def unpack_and_catalog_chandrayaan_data(vault: SouthPoleVault | None = None) -> dict:
    """
    Auto-unzips any Chandrayaan-2 / Chandrayaan-3 / ISRO .zip archives inside
    `datasets/`, `database-chandrayaan-3/`, or `isro_data/`, parses PDS4 .xml labels,
    and registers every file into SQLite `uploaded_datasets`.
    """
    workspace = Path(__file__).parent.resolve()
    search_roots = [
        workspace / "database-chandrayaan-3",
        workspace / "datasets",
        workspace / "isro_data",
    ]
    extracted_dir = workspace / "datasets" / "unpacked_archive"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    unzipped_count = 0
    for root in search_roots:
        if not root.exists():
            continue
        for zfile in root.rglob("*.zip"):
            target = extracted_dir / zfile.stem
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zfile, "r") as zf:
                    zf.extractall(target)
                unzipped_count += 1

    # Parse PDS4 XML metadata map if available
    xml_meta = {}
    for root in search_roots:
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
    ext_counts = {}
    valid_exts = {".csv", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit", ".xml"}
    for root in search_roots:
        if not root.exists():
            continue
        for fpath in root.rglob("*"):
            if fpath.is_file() and fpath.suffix.lower() in valid_exts:
                ext = fpath.suffix.lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
                p_low = str(fpath).lower()
                mission = "Chandrayaan-2" if "chandrayaan_2" in p_low or "ch2" in p_low else (
                    "Chandrayaan-3" if "chandrayaan_3" in p_low or "ch3" in p_low or "navcam" in p_low else "ISRO-Portal"
                )
                inst, lid = xml_meta.get(fpath.stem, ("NavCam/DEM" if ext in {".png", ".tif", ".tiff"} else ext.upper(), "N/A"))
                if "chaste" in p_low:
                    inst = "ChaSTE Thermal Probe"
                elif "libs" in p_low:
                    inst = "LIBS Laser Spectrometer"
                elif "navcam" in p_low:
                    inst = "Pragyan NavCam"
                elif ext in {".fits", ".fit"}:
                    inst = "CH-2 CLASS X-Ray"

                if vault is not None:
                    vault.register_dataset_file(mission, fpath, instrument=inst, pds4_lid=lid)
                cataloged += 1

    return {"unzipped_archives": unzipped_count, "cataloged_files": cataloged, "by_extension": ext_counts}


def _load_raster_normalized(file_path: Path, size: int = 64) -> np.ndarray:
    """Loads ISRO GeoTIFF, NavCam PNG/JPG, or Chandrayaan-2 FITS and normalizes to [0, 1]."""
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
            img = Image.fromarray(arr).resize((size, size))
            arr = np.asarray(img, dtype=np.float32)
    else:
        img = Image.open(file_path)
        if img.mode not in ("L", "F", "I;16"):
            img = img.convert("L")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32)

    # Mask common DEM NoData borders (-9999 / -32768)
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
    return np.column_stack([
        dem.ravel(),
        slope_deg.ravel(),
        thermal_k.ravel(),
        roughness.ravel(),
    ])


def train_from_image_folder(folder_path: str = "isro_data") -> dict:
    """
    Recursively trains the Lunar South Pole AI across `isro_data/`, `datasets/`,
    and `database-chandrayaan-3/` (supporting .tif, .png, .jpg, .fits, and ChaSTE/LIBS .csv).
    """
    vault = SouthPoleVault()
    unpack_and_catalog_chandrayaan_data(vault)

    roots = [Path(folder_path), Path("datasets"), Path("database-chandrayaan-3")]
    valid_img_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if HAS_ASTROPY:
        valid_img_exts.update({".fits", ".fit"})

    rasters, csv_files = [], []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()))
                if p.suffix.lower() in valid_img_exts:
                    rasters.append(p)
                elif p.suffix.lower() == ".csv":
                    csv_files.append(p)

    if not rasters and not csv_files:
        return train_default_southpole_model()

    X_list, y_list = [], []
    sources_used = set()

    for raster_file in rasters[:40]:
        sources_used.add(raster_file.parent.name)
        arr = _load_raster_normalized(raster_file, size=64)
        temp_k = 40.0 + (arr ** 1.4) * 220.0
        feats = extract_features_from_dem(arr, temp_k)

        for elev, slope, temp, rough in feats:
            # Pragyan max operational tilt = 25 deg; Hazard threshold = 20 deg
            if slope > 20.0:
                lbl = "CRATER_WALL_HAZARD"
            elif temp < 105.0:
                # Scientific integrity rule: classify thermal depressions as Candidate PSR Cold Traps
                lbl = "CANDIDATE_PSR_COLD_TRAP"
            elif elev > 0.75:
                lbl = "SUNLIT_PEAK_COMMS"
            else:
                lbl = "SAFE_REGOLITH"
            X_list.append([elev, slope, temp, rough])
            y_list.append(lbl)

    req_cols = {"elevation", "slope_deg", "temp_k", "roughness", "label"}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if req_cols.issubset(set(df.columns)):
                sources_used.add(csv_file.parent.name)
                X_list.extend(df[["elevation", "slope_deg", "temp_k", "roughness"]].values.tolist())
                y_list.extend(df["label"].astype(str).tolist())
        except Exception:
            continue

    if not X_list:
        return train_default_southpole_model()

    source_summary = f"ISRO_CH2_CH3:{','.join(sorted(sources_used))} ({len(rasters)} rasters, {len(csv_files)} csvs)"
    return _fit_and_save(np.array(X_list), np.array(y_list), source=source_summary)


def train_default_southpole_model() -> dict:
    rng = np.random.default_rng(2026)
    n = 2500
    elev = rng.uniform(0.0, 1.0, n)
    slope = rng.uniform(0.0, 42.0, n)
    temp = 40.0 + (elev ** 1.4) * 220.0 + rng.normal(0, 6.0, n)
    rough = slope / 24.0 + rng.normal(0, 0.04, n)

    X = np.column_stack([elev, slope, temp, rough])
    y = []
    for e, s, t in zip(elev, slope, temp):
        if s > 20.0:
            y.append("CRATER_WALL_HAZARD")
        elif t < 105.0:
            y.append("CANDIDATE_PSR_COLD_TRAP")
        elif e > 0.75:
            y.append("SUNLIT_PEAK_COMMS")
        else:
            y.append("SAFE_REGOLITH")

    return _fit_and_save(X, np.array(y), source="CH3_SHIV_SHAKTI_PHYSICS_MODEL")


def _fit_and_save(X: np.ndarray, y: np.ndarray, source: str) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, clf.predict(X_test)))

    bundle = {"model": clf, "accuracy": round(acc * 100.0, 2), "source": source}
    joblib.dump(bundle, MODEL_PATH)
    return {"accuracy_pct": bundle["accuracy"], "source": source, "model_path": str(MODEL_PATH)}


def load_or_train_model():
    if not MODEL_PATH.exists():
        train_default_southpole_model()
    return joblib.load(MODEL_PATH)