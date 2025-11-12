import os
import sys
import logging
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ML.ev_selector import select_threshold_by_ev_unificado

_DOTENV_LOADED = False
try:
    from dotenv import load_dotenv

    load_dotenv()
    _DOTENV_LOADED = True
except Exception:
    _DOTENV_LOADED = False


def _load_env_file():
    candidates = [
        os.getenv("MODELML_ENV_FILE"),
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for path in candidates:
        if not path:
            continue
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            continue
        break


_load_env_file()

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

import joblib
import optuna
from optuna.exceptions import TrialPruned
import json

def _get_bool(name: str, default: bool) -> bool:
    """Get a boolean value from an environment variable."""
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip() in {"1", "true", "True", "YES", "yes"}

def _get_int(name: str, default: int) -> int:
    """Get an integer value from an environment variable."""
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    """Get a float value from an environment variable."""
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_PATH, "data")
DEFAULT_DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "BTCUSDT_15m_processed.csv")
ALT_DATA_PATH     = os.path.join(BASE_PATH, "data", "BTCUSDT_15m_processed.csv")
DATA_PATH_ENV     = os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
ALT_DATA_PATH_ENV = os.getenv("ALT_DATA_PATH", ALT_DATA_PATH)
CSV_PATH = DATA_PATH_ENV if os.path.exists(DATA_PATH_ENV) else ALT_DATA_PATH_ENV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró DATA_PATH ni ALT_DATA_PATH: {DATA_PATH_ENV} | {ALT_DATA_PATH_ENV}")

MODEL_DIR    = os.getenv("MODEL_DIR", os.path.join(BASE_PATH, "results"))
RESULTS_PATH = MODEL_DIR
os.makedirs(RESULTS_PATH, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "XGBoost_Binario15m")


TEST_SIZE       = _get_float("TEST_SIZE", 0.3)
RANDOM_SEED     = _get_int("RANDOM_SEED", 44)

SAMPLE_FRAC     = _get_float("SAMPLE_FRAC", 0.90)
N_TRIALS        = _get_int("N_TRIALS", 55)
N_JOBS_OPTUNA   = _get_int("N_JOBS_OPTUNA", 1)
OPTUNA_TIMEOUT  = _get_int("OPTUNA_TIMEOUT", 2400)  # s

USE_PCA          = _get_bool("USE_PCA", False)
PCA_N_COMPONENTS = _get_int("PCA_N_COMPONENTS", 8)

USE_GPU = _get_bool("USE_GPU", False)
TREE_METHOD = "gpu_hist" if USE_GPU else "hist"
GENERATE_PORTFOLIO_REPORT   = _get_bool("GENERATE_PORTFOLIO_REPORT", True)
GENERATE_PROBABILITY_PLOTS  = _get_bool("GENERATE_PROBABILITY_PLOTS", True)
GENERATE_FEATURE_IMPORTANCE = _get_bool("GENERATE_FEATURE_IMPORTANCE", True)
PERM_IMPORTANCE_SAMPLES     = _get_int("PERM_IMPORTANCE_SAMPLES", 4000)
PERM_IMPORTANCE_REPEATS     = _get_int("PERM_IMPORTANCE_REPEATS", 5)
THRESHOLD_SWEEP_POINTS      = max(10, _get_int("THRESHOLD_SWEEP_POINTS", 40))
PORTFOLIO_REPORT_NAME       = os.getenv("PORTFOLIO_REPORT_NAME", "training_report.md")


look_ahead_cfg = int(os.getenv("LOOK_AHEAD", 3))
if int(os.getenv("LOCK_LOOKAHEAD_3", 1)) == 1 and look_ahead_cfg != 3:
    logging.info("LOOK_AHEAD=%s recibido, pero LOCK_LOOKAHEAD_3=1 => forzando LOOK_AHEAD=3.", look_ahead_cfg)
    look_ahead_cfg = 3
FIXED_LOOK_AHEAD   = look_ahead_cfg
FIXED_MIN_CHANGE   = _get_float("FIXED_MIN_CHANGE", 0.031)
USE_DYNAMIC_LABEL  = _get_bool("USE_DYNAMIC_LABEL", True)
VOL_WINDOW         = _get_int("VOL_WINDOW", 30)
K_VOL              = _get_float("K_VOL", 1.5)


TP_MULT           = _get_float("TP_MULT", 1.2)
SL_MULT           = _get_float("SL_MULT", 0.4)
COMMISSION_RATE   = _get_float("COMMISSION_RATE", 0.0003)     # por lado
BASE_SPREAD_PCT   = _get_float("SPREAD_PCT", 0.00005)
BASE_SLIPPAGE_PCT = _get_float("SLIPPAGE_BASE", 0.0001)
SLIP_MAX_PCT      = _get_float("SLIP_MAX_PCT", 0.0010)
SLIP_RANGE_COEF   = _get_float("SLIP_RANGE_COEF", 0.10)

MIN_THR_FRAC_EV   = _get_float("MIN_THR_FRAC", 0.0025)        # MISMO que backtest
LABEL_MIN_MOVE_BPS = _get_float("LABEL_MIN_MOVE_BPS", 0.0)
LABEL_MIN_MOVE_FRAC = (LABEL_MIN_MOVE_BPS / 10000.0) if LABEL_MIN_MOVE_BPS else 0.0
MIN_THR_FRAC_FLOOR = max(MIN_THR_FRAC_EV, LABEL_MIN_MOVE_FRAC)
RISK_PER_TRADE_EV = _get_float("RISK_PER_TRADE", 0.0040)      # informativo (se cancela en EV relativo)

EV_GRID_LOW       = _get_float("EV_GRID_LOW", 0.55)
EV_GRID_HIGH      = _get_float("EV_GRID_HIGH", 0.80)
EV_GRID_POINTS    = max(5, _get_int("EV_GRID_POINTS", 50))
EV_QS_GRID        = np.linspace(EV_GRID_LOW, EV_GRID_HIGH, EV_GRID_POINTS)

MIN_SIGNALS_EV    = _get_int("MIN_SIGNALS_EV", 90)
EV_MEAN_MIN       = _get_float("EV_MEAN_MIN", 0.015)
EV_MARGIN         = _get_float("EV_MARGIN", 0.005)
PROB_THRESHOLD_FALLBACK = _get_float("PROB_THRESHOLD_FALLBACK", 0.80)
THR_BASE_DEFAULT  = _get_float("THR_BASE_DEFAULT", 0.50)
THR_MAX_CAP       = _get_float("THR_MAX_CAP", 0.82)
MIN_TRADES_DAILY_TARGET = _get_float("MIN_TRADES_DAILY", 1.8)
MAX_TRADES_DAILY_TARGET = _get_float("MAX_TRADES_DAILY", 2.5)
THR_HOLDOUT_FRAC  = float(np.clip(_get_float("THR_HOLDOUT_FRAC", 0.55), 0.2, 0.8))
MIN_CALIBRATION_SIZE = _get_int("MIN_CALIBRATION_SIZE", 120)


ISO_DEFAULT_PATH = os.path.join(MODEL_DIR, "iso_cal.joblib")
if os.path.exists(ISO_DEFAULT_PATH):
    PROB_CLIP_LOW = 0.0
    PROB_CLIP_HIGH = 1.0
    PROB_TEMPERATURE = 1.0
else:
    PROB_CLIP_LOW   = _get_float("PROB_CLIP_LOW", 0.0)
    PROB_CLIP_HIGH  = _get_float("PROB_CLIP_HIGH", 1.0)
    PROB_TEMPERATURE= _get_float("PROB_TEMPERATURE", 1.0)

USE_TREND_FILTER   = _get_bool("USE_TREND_FILTER", True)
USE_EMA200_FILTER  = _get_bool("USE_EMA200_FILTER", True)
MIN_RANGE_C_ENTRY  = _get_float("MIN_RANGE_C_ENTRY", 0.0030)
ADX_MIN            = _get_float("ADX_MIN", 25.0)
TRADING_WINDOWS    = os.getenv("TRADING_WINDOWS", "00:00-23:59").strip()


def setup_logging(log_path=os.path.join(BASE_PATH, "logs")):
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger("training_binario_ev")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(os.path.join(log_path, f"training_{timestamp}.log"), mode="a")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

from ML.features import add_derived_features, get_feature_cols

logger = setup_logging()


def load_data(csv_path: str) -> pd.DataFrame:
    # Carga robusta del dataset asegurando índices temporales limpios y columnas clave.
    required_price_cols = {"close", "high", "low", "volume", "open"}
    candidates = []
    if csv_path:
        candidates.append(csv_path)
    if csv_path != DATA_PATH_ENV:
        candidates.append(DATA_PATH_ENV)
    if csv_path != ALT_DATA_PATH_ENV:
        candidates.append(ALT_DATA_PATH_ENV)
    path = None
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        logger.error(f"No se encontró archivo de datos en {candidates}")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        time_col = "time" if "time" in df.columns else "open_time"
        if time_col not in df.columns:
            raise KeyError("Debe existir columna 'time' u 'open_time' en el dataset procesado.")
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.dropna(subset=[time_col], inplace=True)
        df.drop_duplicates(subset=[time_col], inplace=True)
        df.sort_values(by=time_col, inplace=True)
        df.set_index(time_col, inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        missing = required_price_cols - set(df.columns)
        if missing:
            raise KeyError(f"Faltan columnas imprescindibles: {missing}")
        df = df.sort_index()
        n_total = len(df)
        n_expected = max(0, n_total - FIXED_LOOK_AHEAD)
        logger.info(
            "[SANITY] Datos cargados desde %s | filas=%d | post-lookahead≈%d",
            path,
            n_total,
            n_expected,
        )
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        sys.exit(1)


def infer_bars_per_day_from_index(idx: pd.Index) -> int:
    default_bpd = int(os.getenv("BARS_PER_DAY", 96))
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return default_bpd
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return default_bpd
    median_delta = diffs.median()
    if pd.isna(median_delta) or median_delta.total_seconds() <= 0:
        return default_bpd
    minutes = max(1, int(round(median_delta.total_seconds() / 60.0)))
    bars_per_day = max(1, int(round(1440 / minutes)))
    return bars_per_day


def generate_target_binary(df: pd.DataFrame,
                           look_ahead: int = FIXED_LOOK_AHEAD,
                           min_change: float = FIXED_MIN_CHANGE) -> pd.DataFrame:
    # Construye la etiqueta binaria respetando look-ahead y umbrales dinámicos basados en volatilidad.
    if look_ahead <= 0:
        logger.error("look_ahead debe ser positivo.")
        sys.exit(1)
    d = df.copy()
    d["future_close"] = d["close"].shift(-look_ahead)
    d.dropna(subset=["future_close"], inplace=True)
    d["change_pct"] = (d["future_close"] - d["close"]) / d["close"]
    ret1 = d["close"].pct_change(1, fill_method=None)
    if USE_DYNAMIC_LABEL:
        if "ATR_14" not in d.columns:
            prev_close = d["close"].shift(1)
            tr = pd.concat(
                [
                    (d["high"] - d["low"]).abs(),
                    (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            d["ATR_14"] = tr.rolling(14, min_periods=14).mean()
        atr_frac = (d["ATR_14"] / d["close"]).replace([np.inf, -np.inf], np.nan)
        atr_frac = atr_frac.fillna(method="ffill").fillna(method="bfill")
        d["thr_frac"] = (K_VOL * atr_frac).clip(lower=MIN_THR_FRAC_FLOOR).fillna(MIN_THR_FRAC_FLOOR)
        buy  = d["change_pct"] >= d["thr_frac"]
        sell = d["change_pct"] <= -d["thr_frac"]
    else:
        if not (0 < min_change):
            logger.error("min_change debe ser positivo si USE_DYNAMIC_LABEL=False.")
            sys.exit(1)
        buy  = d["change_pct"] >= min_change
        sell = d["change_pct"] <= -min_change
        d["thr_frac"] = float(max(min_change, MIN_THR_FRAC_FLOOR))
    d["target"] = np.where(buy, 1, np.where(sell, 0, np.nan))
    d.dropna(subset=["change_pct", "target"], inplace=True)
    vc = d['target'].value_counts().to_dict()
    logger.info(f"Target binario generado. Dist. clases (0/1): {vc}")
    return d





def postprocess_probs(probs: np.ndarray) -> np.ndarray:
    if PROB_CLIP_LOW <= 0.0 and PROB_CLIP_HIGH >= 1.0 and abs(PROB_TEMPERATURE - 1.0) < 1e-6:
        return np.asarray(probs, dtype=float)
    eps = 1e-3
    low = PROB_CLIP_LOW if PROB_CLIP_LOW > 0.0 else eps
    high = PROB_CLIP_HIGH if PROB_CLIP_HIGH < 1.0 else 1.0 - eps
    probs = np.clip(probs, low, high)
    z = np.log(probs / (1 - probs + 1e-12))
    z = z / max(PROB_TEMPERATURE, 1e-6)
    out = 1.0 / (1.0 + np.exp(-z))
    return np.clip(out, PROB_CLIP_LOW, PROB_CLIP_HIGH)

def _parse_windows(w: str):
    items = []
    for part in w.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        h1, m1 = map(int, a.split(":"))
        h2, m2 = map(int, b.split(":"))
        items.append((h1 * 60 + m1, h2 * 60 + m2))
    return items

def time_window_mask(idx: pd.Index, wins_str: str) -> pd.Series:
    wins = _parse_windows(wins_str)
    if not wins:
        return pd.Series(True, index=idx)
    minutes = (idx.hour * 60 + idx.minute).astype(int)
    mask = np.zeros(len(idx), dtype=bool)
    for start, end in wins:
        mask |= (minutes >= start) & (minutes <= end)
    return pd.Series(mask, index=idx)

from ML.indicators import compute_adx, compute_atr

def add_trade_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA_50"] = d["close"].rolling(50, min_periods=50).mean()
    d["SMA_200"] = d["close"].rolling(200, min_periods=200).mean()
    d["EMA_10"] = d["close"].ewm(span=10, adjust=False).mean()
    d["EMA_50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["EMA_200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["SMA200_slope"] = d["SMA_200"] - d["SMA_200"].shift(10)
    d["ATR"] = compute_atr(d, 14)
    atr_frac = (d["ATR"] / d["close"].replace(0, np.nan)).fillna(0.0)
    d["thr_frac"] = (K_VOL * atr_frac).clip(lower=MIN_THR_FRAC_FLOOR).fillna(MIN_THR_FRAC_FLOOR)
    d["range_c"] = (d["high"] - d["low"]).abs() / d["close"]
    d["ADX"] = compute_adx(d, 14)
    d["in_window"] = time_window_mask(d.index, TRADING_WINDOWS)
    return d

def entry_mask(df: pd.DataFrame, side: str) -> pd.Series:
    if USE_TREND_FILTER:
        if side == "long":
            cond = df["EMA_10"] > df["EMA_50"]
            if USE_EMA200_FILTER:
                cond &= (df["EMA_50"] >= df["EMA_200"])
        else:
            cond = df["EMA_10"] < df["EMA_50"]
            if USE_EMA200_FILTER:
                cond &= (df["EMA_50"] <= df["EMA_200"])
    else:
        cond = pd.Series(True, index=df.index)
    cond &= (df["range_c"] >= MIN_RANGE_C_ENTRY)
    cond &= (df["ADX"] >= ADX_MIN)
    cond &= df["in_window"].astype(bool)
    return cond.astype(bool)


def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "target",
    label_candidates: Optional[list[str]] = None,
):
    # Selecciona columnas finales, asegura la etiqueta y registra cualquier feature faltante.
    df = df.copy()
    keep = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Features faltantes (se imputarán en el Pipeline): {missing}")
    label_candidates = label_candidates or []
    target_col = label_col if label_col in df.columns else None
    if target_col is None:
        for alt in label_candidates:
            if alt in df.columns:
                target_col = alt
                if alt != label_col:
                    logger.info("Usando columna de etiqueta alternativa '%s'.", alt)
                break
    cols = keep + ([target_col] if target_col else [])
    df = df.loc[:, [c for c in cols if c in df.columns]]
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
    else:
        logger.warning(
            "No se encontró ninguna columna de etiqueta (%s + %s). Se usarán ceros temporales.",
            label_col,
            label_candidates,
        )
        X = df
        y = pd.Series([0] * len(df), index=df.index, dtype=int)
    if not all(c in X.columns for c in keep):
        logger.error("Columnas de features inconsistentes.")
        sys.exit(1)
    logger.info(f"Dataset preparado: {X.shape[0]} filas, {X.shape[1]} features.")
    return X, y

# ============ PURGED + EMBARGO TIME SERIES SPLIT (evita fuga temporal) ======
class EmbargoedPurgedSplit:
    def __init__(self, n_splits=5, purge=1, embargo=0, min_train=500):
        # División temporal con purga/embargo para evitar fuga de información.
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.min_train = min_train

    def split(self, X, y=None):
        n = len(X)
        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(fold_sizes) - fold_sizes
        ends = np.cumsum(fold_sizes)

        for i in range(self.n_splits):
            val_start, val_end = int(starts[i]), int(ends[i])
            train_end = max(0, val_start - self.purge)

            left_idx = indices[:train_end]
            embargo_start = min(n, val_end + max(0, self.embargo))
            right_idx = indices[embargo_start:]
            train_idx = np.concatenate([left_idx, right_idx]) if len(right_idx) else left_idx

            val_idx = indices[val_start:val_end]
            if len(train_idx) < self.min_train or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

# ============================ (Opcional) SMOTE ===============================
def balance_classes_smote(X, y):
    # Sobre-muestreo opcional para combatir desbalance de clases.
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imblearn no instalado; SMOTE desactivado.")
        return X, y
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def compute_scale_pos_weight(y, factor_pos_weight: float = 1.0) -> float:
    """
    Calcula n_neg/n_pos (con guardas) y aplica un factor opcional.
    """
    y_arr = np.asarray(y).astype(int)
    counts = np.bincount(y_arr, minlength=2)
    n_neg = max(1, counts[0])
    n_pos = max(1, counts[1])
    if counts[0] == 0 or counts[1] == 0:
        logger.warning("compute_scale_pos_weight: distribución degenerada (neg=%d, pos=%d).", counts[0], counts[1])
    return (n_neg / n_pos) * max(1e-3, float(factor_pos_weight))


def build_preprocessing_steps(use_pca: bool, pca_n_components: int):
    # Armado del pipeline de preprocesamiento (imputación, escalado y PCA opcional).
    steps = [("imputer", SimpleImputer(strategy="mean")),
             ("scaler", RobustScaler())]
    if use_pca:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=RANDOM_SEED)))
    return steps


# ============================ OPTUNA (PR-AUC + PRUNING) =====================
def objective_pr_auc(trial, X, y, cv):
    # Objetivo de Optuna: maximizar PR-AUC bajo CV temporal con poda temprana.
    scale_pos_weight = compute_scale_pos_weight(y)
    param = {
        "max_depth": trial.suggest_int("max_depth", 4, 11),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 320, 1200),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 4, 18),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 6),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        "tree_method": TREE_METHOD,
        "n_jobs": 0,
        "scale_pos_weight": scale_pos_weight,
    }
    pipeline_steps = build_preprocessing_steps(USE_PCA, PCA_N_COMPONENTS)
    pipeline_steps.append(("model", XGBClassifier(**param)))
    pipeline = Pipeline(pipeline_steps)

    scores = []
    valid_folds = 0
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        if os.getenv("USE_SMOTE", "0") == "1":
            X_train, y_train = balance_classes_smote(X_train, y_train)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_valid)[:, 1]
        pr_auc = average_precision_score(y_valid, y_prob)
        scores.append(pr_auc)
        trial.report(float(np.mean(scores)), step=fold_idx)
        if trial.should_prune():
            raise TrialPruned()
        valid_folds += 1
    if valid_folds == 0:
        return -np.inf
    return float(np.mean(scores))

def train_models_optuna_pr_auc(X, y, cv,
                               n_trials=N_TRIALS,
                               n_jobs=N_JOBS_OPTUNA,
                               timeout=OPTUNA_TIMEOUT):
    logger.info(f"Optuna: PR-AUC con CV temporal | trials={n_trials} | n_jobs={n_jobs} | timeout={timeout}s")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=48),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    study.optimize(lambda t: objective_pr_auc(t, X, y, cv),
                   n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)
    logger.info(f"Mejor PR-AUC: {study.best_value:.4f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    return study.best_params

# =================== ENTRENAR PIPELINE FINAL (con params) ====================
def train_final_pipeline_with_params(X, y, best_params, use_pca=False, pca_n_components=7,
                                     factor_pos_weight=1.0):
    # Entrena el pipeline definitivo con los hiperparámetros óptimos y ponderación de clases.
    try:
        steps = build_preprocessing_steps(use_pca, pca_n_components)
        scale_pos_weight = compute_scale_pos_weight(y, factor_pos_weight)
        model = XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_SEED,
            eval_metric="aucpr",
            verbosity=0,
            tree_method=TREE_METHOD,
            n_jobs=0,
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            gamma=best_params["gamma"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            min_child_weight=best_params["min_child_weight"],
            max_delta_step=best_params["max_delta_step"],
            scale_pos_weight=scale_pos_weight
        )
        steps.append(("model", model))
        pipeline = Pipeline(steps)
        pipeline.fit(X, y)
        logger.info(f"Pipeline final entrenado (scale_pos_weight={scale_pos_weight:.2f})")
        return pipeline
    except Exception as e:
        logger.error(f"Error al entrenar el pipeline final: {e}")
        sys.exit(1)

# ====================== EVALUACIÓN Y GRÁFICOS (TEST) ========================
def evaluate_and_plots(model, X, y, y_prob, best_threshold, results_path=RESULTS_PATH, title_suffix=""):
    # Evalúa el set objetivo y genera los gráficos/artefactos mínimos para portafolio.
    artifact_paths = []
    y_int = y.astype(int).values
    auc   = roc_auc_score(y_int, y_prob)
    ap    = average_precision_score(y_int, y_prob)
    brier = brier_score_loss(y_int, y_prob)

    y_pred_thr = (y_prob >= best_threshold).astype(int)
    prec_thr = precision_score(y_int, y_pred_thr, pos_label=1, zero_division=0)
    rec_thr  = recall_score(y_int, y_pred_thr, pos_label=1, zero_division=0)
    f1_thr   = f1_score(y_int, y_pred_thr, pos_label=1, zero_division=0)
    bacc_thr = balanced_accuracy_score(y_int, y_pred_thr)
    mcc_thr  = matthews_corrcoef(y_int, y_pred_thr)
    base_pos = y_int.mean()

    logger.info(f"[Eval{title_suffix} con thr={best_threshold:.3f}] AUC={auc:.4f} | PR-AUC={ap:.4f} | Brier={brier:.4f} | "
                f"Prec={prec_thr:.4f} | Recall={rec_thr:.4f} | F1={f1_thr:.4f} | "
                f"BAcc={bacc_thr:.4f} | MCC={mcc_thr:.4f} | PosRate={base_pos:.3f}")

    metrics = {
        "auc": auc,
        "pr_auc": ap,
        "brier_score": brier,
        "threshold": best_threshold,
        "precision": prec_thr,
        "recall": rec_thr,
        "f1_score": f1_thr,
        "balanced_accuracy": bacc_thr,
        "mcc": mcc_thr,
        "positive_rate": base_pos,
    }
    metrics_path = os.path.join(results_path, f"metrics{title_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    artifact_paths.append(metrics_path)
    logger.info(f"Métricas de evaluación guardadas en metrics{title_suffix}.json")

    # Matriz de Confusión
    cm = confusion_matrix(y_int, y_pred_thr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Neg", "Pred Pos"], yticklabels=["Real Neg", "Real Pos"])
    plt.title(f"Matriz de Confusión{title_suffix} (Threshold = {best_threshold:.3f})")
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.tight_layout()
    cm_path = os.path.join(results_path, f"confusion_matrix{title_suffix}.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Matriz de confusión guardada en confusion_matrix{title_suffix}.png")
    artifact_paths.append(cm_path)

    # PR curve
    p, r, _ = precision_recall_curve(y_int, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.title(f"Precision-Recall Curve{title_suffix} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True); plt.tight_layout()
    pr_path = os.path.join(results_path, f"pr_curve_binary{title_suffix}.png")
    plt.savefig(pr_path)
    plt.close()
    artifact_paths.append(pr_path)

    # ROC
    fpr, tpr, _ = roc_curve(y_int, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title(f"Curva ROC (Binaria){title_suffix}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.grid(True); plt.tight_layout()
    roc_path = os.path.join(results_path, f"roc_curve_binary{title_suffix}.png")
    plt.savefig(roc_path)
    plt.close()
    artifact_paths.append(roc_path)

    # Reliability diagram (calibración)
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_int, y_prob, n_bins=12, strategy="quantile")
        try:
            rel_suffix = "_test" if title_suffix == "_test" else title_suffix
            rel_name = "reliability_test.csv" if rel_suffix == "_test" else f"reliability{rel_suffix}.csv"
            rel_path = os.path.join(MODEL_DIR, rel_name.lstrip("_"))
            pd.DataFrame({"prob_pred": mean_pred, "prob_true": frac_pos}).to_csv(rel_path, index=False)
            logger.info("Curva de confiabilidad guardada en %s", rel_path)
            artifact_paths.append(rel_path)
        except Exception as rel_exc:
            logger.warning("No se pudo guardar reliability csv: %s", rel_exc)
        plt.figure()
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0,1],[0,1],"k--")
        plt.title(f"Calibration (Reliability) Curve{title_suffix}\nBrier={brier:.3f}")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.grid(True); plt.tight_layout()
        cal_path = os.path.join(results_path, f"calibration_curve{title_suffix}.png")
        plt.savefig(cal_path)
        plt.close()
        artifact_paths.append(cal_path)
    except Exception as e:
        logger.warning(f"No se pudo generar curva de calibración: {e}")

    # Classification report (texto)
    try:
        cls_rep = classification_report(y_int, y_pred_thr, digits=4, zero_division=0)
        cls_path = os.path.join(results_path, f"classification_report{title_suffix}.txt")
        with open(cls_path, "w") as f:
            f.write(cls_rep)
        artifact_paths.append(cls_path)
        logger.info("Classification report guardado en %s", cls_path)
    except Exception as rep_exc:
        logger.warning("No se pudo guardar classification_report: %s", rep_exc)

    # Histogramas y barridos de threshold para portafolio
    if GENERATE_PROBABILITY_PLOTS:
        try:
            plt.figure()
            bins = np.linspace(0, 1, 30)
            plt.hist(y_prob[y_int == 1], bins=bins, alpha=0.6, label="Clase 1")
            plt.hist(y_prob[y_int == 0], bins=bins, alpha=0.6, label="Clase 0")
            plt.axvline(best_threshold, color="red", linestyle="--", label=f"Threshold={best_threshold:.3f}")
            plt.title(f"Distribución de probabilidades{title_suffix}")
            plt.xlabel("Probabilidad"); plt.ylabel("Frecuencia")
            plt.legend()
            plt.tight_layout()
            hist_path = os.path.join(results_path, f"probability_hist{title_suffix}.png")
            plt.savefig(hist_path)
            plt.close()
            artifact_paths.append(hist_path)
        except Exception as hist_exc:
            logger.warning("No se pudo generar histograma de probabilidades: %s", hist_exc)

        try:
            thr_values = np.linspace(0.05, 0.95, THRESHOLD_SWEEP_POINTS)
            sweep_rows = []
            precision_vals, recall_vals = [], []
            for thr in thr_values:
                preds = (y_prob >= thr).astype(int)
                precision_vals.append(precision_score(y_int, preds, zero_division=0))
                recall_vals.append(recall_score(y_int, preds, zero_division=0))
                sweep_rows.append(
                    {
                        "threshold": round(float(thr), 6),
                        "precision": precision_vals[-1],
                        "recall": recall_vals[-1],
                        "f1": f1_score(y_int, preds, zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_int, preds),
                    }
                )
            sweep_df = pd.DataFrame(sweep_rows)
            sweep_path = os.path.join(results_path, f"threshold_sweep{title_suffix}.csv")
            sweep_df.to_csv(sweep_path, index=False)
            artifact_paths.append(sweep_path)

            plt.figure()
            plt.plot(thr_values, precision_vals, label="Precision")
            plt.plot(thr_values, recall_vals, label="Recall")
            plt.axvline(best_threshold, color="red", linestyle="--", label="Threshold óptimo")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Precision/Recall por Threshold{title_suffix}")
            plt.grid(True)
            plt.legend()
            thr_plot_path = os.path.join(results_path, f"threshold_precision_recall{title_suffix}.png")
            plt.tight_layout()
            plt.savefig(thr_plot_path)
            plt.close()
            artifact_paths.append(thr_plot_path)
        except Exception as sweep_exc:
            logger.warning("No se pudo generar threshold sweep: %s", sweep_exc)

    return metrics, artifact_paths

# ============================ GUARDAR =======================================
def compute_feature_importance_artifacts(pipeline, X_ref, y_ref, results_path, suffix="_test"):
    # Calcula permutation importance para documentar drivers del modelo.
    artifacts = []
    try:
        if len(X_ref) == 0:
            return artifacts
        sample_size = min(len(X_ref), max(200, PERM_IMPORTANCE_SAMPLES))
        if len(X_ref) > sample_size:
            X_sample = X_ref.sample(sample_size, random_state=RANDOM_SEED)
        else:
            X_sample = X_ref.copy()
        y_sample = y_ref.loc[X_sample.index]
        importance = permutation_importance(
            pipeline,
            X_sample,
            y_sample.astype(int),
            scoring="average_precision",
            n_repeats=max(3, PERM_IMPORTANCE_REPEATS),
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        imp_df = pd.DataFrame(
            {
                "feature": list(X_sample.columns),
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        imp_df["importance_abs"] = imp_df["importance_mean"].abs()
        imp_df = imp_df.sort_values("importance_abs", ascending=False)
        csv_path = os.path.join(results_path, f"feature_importance_permutation{suffix}.csv")
        imp_df.to_csv(csv_path, index=False)
        artifacts.append(csv_path)

        top_n = min(20, len(imp_df))
        if top_n > 0:
            top_df = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal plot
            plt.figure(figsize=(10, max(4, 0.4 * top_n)))
            plt.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"])
            plt.xlabel("Permutation Importance (mean ΔAP)")
            plt.title(f"Top {top_n} Features (Permutation Importance){suffix}")
            plt.tight_layout()
            fi_path = os.path.join(results_path, f"feature_importance_plot{suffix}.png")
            plt.savefig(fi_path)
            plt.close()
            artifacts.append(fi_path)
        logger.info("Permutation importance calculado en %s", csv_path)
    except Exception as imp_exc:
        logger.warning("No se pudo calcular permutation importance: %s", imp_exc)
    return artifacts


def _safe_isoformat(ts):
    if ts is None:
        return "NA"
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def summarize_split(df: pd.DataFrame, label: str, target_col: str = "target") -> dict:
    # Resumen por bloque temporal: filas, periodo y balance de clases.
    summary = {
        "label": label,
        "rows": int(len(df)),
        "start": _safe_isoformat(df.index.min() if len(df) else None),
        "end": _safe_isoformat(df.index.max() if len(df) else None),
        "missing_rate": float(df.isna().mean().mean()) if len(df.columns) else 0.0,
    }
    if target_col in df.columns:
        tgt = df[target_col].dropna()
        summary["positive_rate"] = float(tgt.mean()) if not tgt.empty else float("nan")
        summary["class_counts"] = {int(k): int(v) for k, v in tgt.value_counts().to_dict().items()}
    else:
        summary["positive_rate"] = float("nan")
        summary["class_counts"] = {}
    return summary


def generate_training_report(results_path, model_name, dataset_summary, best_params,
                             best_threshold, metrics, optuna_cfg, calibration_method,
                             artifact_paths):
    # Reporte Markdown amigable para portafolio con métricas, hiperparámetros y artefactos.
    report_path = os.path.join(results_path, PORTFOLIO_REPORT_NAME)
    try:
        lines = [
            f"# {model_name} – Training Summary",
            "",
            f"- Fecha de ejecución: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"- Método de calibración: {calibration_method}",
            f"- Threshold seleccionado (EV): {best_threshold:.4f}",
            "",
            "## Dataset",
        ]
        for split_name, summary in dataset_summary.items():
            pos_rate = summary.get("positive_rate")
            pos_txt = f"{pos_rate:.3f}" if pos_rate == pos_rate else "NA"  # NaN check
            lines.append(
                f"- **{split_name.title()}** | filas={summary['rows']} | "
                f"período={summary['start']} → {summary['end']} | pos_rate={pos_txt}"
            )
        lines.extend(
            [
                "",
                "## Métricas (Test)",
                "| Métrica | Valor |",
                "| --- | --- |",
            ]
        )
        for k, v in metrics.items():
            lines.append(f"| {k} | {v:.6f} |")

        lines.extend(
            [
                "",
                "## Hiperparámetros (Optuna)",
                "| Parámetro | Valor |",
                "| --- | --- |",
            ]
        )
        for k, v in best_params.items():
            lines.append(f"| {k} | {v} |")

        lines.extend(
            [
                "",
                "## Configuración de Búsqueda",
                "| Clave | Valor |",
                "| --- | --- |",
            ]
        )
        for k, v in optuna_cfg.items():
            lines.append(f"| {k} | {v} |")

        if artifact_paths:
            lines.extend(
                [
                    "",
                    "## Artefactos Generados",
                ]
            )
            for path in sorted(set(artifact_paths)):
                rel = os.path.relpath(path, results_path)
                lines.append(f"- {rel}")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Reporte de entrenamiento guardado en %s", report_path)
    except Exception as rep_exc:
        logger.error("No se pudo generar training_report: %s", rep_exc)
        return ""
    return report_path


def save_model_and_results(model, best_params, feature_cols, model_name=MODEL_NAME,
                           results_path=RESULTS_PATH, alternative_save_path=None,
                           best_threshold=None, calibrator=None):
    os.makedirs(results_path, exist_ok=True)

    pipeline_dir = os.getenv("MODEL_OUT_DIR", results_path)
    pipeline_name = os.getenv("MODEL_PIPELINE_NAME", f"{model_name}_trained_pipeline.joblib")
    pipeline_path = os.path.join(pipeline_dir, pipeline_name)

    def _dump_pipeline(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    try:
        _dump_pipeline(pipeline_path)
        logger.info("Pipeline guardado en: %s", pipeline_path)
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            logger.error("No hay suficiente espacio en disco para guardar el pipeline en %s.", pipeline_dir)
            if alternative_save_path:
                try:
                    fallback_path = os.path.join(alternative_save_path, pipeline_name)
                    _dump_pipeline(fallback_path)
                    logger.info("Modelo guardado en ubicación alternativa: %s", fallback_path)
                except Exception as ex:
                    logger.error("No se pudo guardar en la ubicación alternativa: %s", ex)
            else:
                logger.error("No se proporcionó ubicación alternativa.")
        else:
            logger.error("Error al guardar el modelo: %s", e)
    with open(os.path.join(results_path, f"{model_name}_model_params.txt"), "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(results_path, f"{model_name}_feature_cols.txt"), "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    if best_threshold is not None:
        with open(os.path.join(results_path, "threshold.txt"), "w") as f:
            f.write(f"{best_threshold:.6f}\n")
    if calibrator is not None:
        try:
            cal_path = os.path.join(results_path, "iso_cal.joblib")
            joblib.dump(calibrator, cal_path)
            logger.info("Calibrador guardado en: %s", cal_path)
        except Exception as cal_exc:
            logger.error("No se pudo guardar el calibrador: %s", cal_exc)
    logger.info(f"Artefactos guardados en {results_path}")

# ============================ LOG ENV (sanity) ===============================
def _log_effective_env():
    logger.info(
        "EV/Costes efectivos → COMMISSION_RATE=%.7f | SLIPPAGE_BASE=%.7f | "
        "SPREAD_PCT=%.7f | SLIP_MAX_PCT=%.7f | SLIP_RANGE_COEF=%.3f | MIN_THR_FRAC=%.5f | "
        "LABEL_MIN_MOVE_BPS=%.2f | THR_FLOOR=%.5f | EV_GRID_LOW=%.2f | EV_GRID_HIGH=%.2f | "
        "EV_GRID_POINTS=%d | MIN_SIGNALS_EV=%d | EV_MEAN_MIN=%.3f | EV_MARGIN=%.3f",
        COMMISSION_RATE, BASE_SLIPPAGE_PCT, BASE_SPREAD_PCT,
        SLIP_MAX_PCT, SLIP_RANGE_COEF, MIN_THR_FRAC_EV,
        LABEL_MIN_MOVE_BPS, MIN_THR_FRAC_FLOOR,
        EV_GRID_LOW, EV_GRID_HIGH, EV_GRID_POINTS, MIN_SIGNALS_EV,
        EV_MEAN_MIN, EV_MARGIN
    )

# ================================= MAIN =====================================
def main():
    # Orquestador principal del entrenamiento: carga datos, entrena, calibra y reporta.
    try:
        # 1) Cargar
        df = load_data(CSV_PATH)
        _log_effective_env()  # sanity-check: confirma que tomó tu .env

        # 2) Target (agrega future_close y thr_frac)
        df = generate_target_binary(df, look_ahead=FIXED_LOOK_AHEAD, min_change=FIXED_MIN_CHANGE)
        if "target" in df.columns and "target_bin" not in df.columns:
            df["target_bin"] = df["target"].astype(int)

        # 3) Split temporal
        split_idx = int(len(df) * (1 - TEST_SIZE))
        df_train = df.iloc[:split_idx].copy()
        df_test  = df.iloc[split_idx:].copy()
        logger.info(f"Split: Train={len(df_train)}, Test={len(df_test)}")
        dataset_summary = {
            "train": summarize_split(df_train, "train"),
            "test": summarize_split(df_test, "test"),
        }
        portfolio_artifacts = []

        # 4) Features
        df_train = add_derived_features(df_train)
        df_test  = add_derived_features(df_test)
        try:
            feature_cols = get_feature_cols(RESULTS_PATH, MODEL_NAME)
            logger.info(f"Cargadas {len(feature_cols)} features desde archivo.")
        except FileNotFoundError as e:
            logger.error(e)
            sys.exit(1)

        # 5) Dataset
        label_fallbacks = ["target_up", "target_bin", "label"]
        X_train, y_train = prepare_dataset(
            df_train,
            feature_cols,
            label_col="target",
            label_candidates=label_fallbacks,
        )
        X_test, y_test = prepare_dataset(
            df_test,
            feature_cols,
            label_col="target",
            label_candidates=label_fallbacks,
        )

        # 6) CV temporal (purge+embargo) y split para calibración hold-out
        tscv = EmbargoedPurgedSplit(n_splits=5, purge=FIXED_LOOK_AHEAD, embargo=FIXED_LOOK_AHEAD, min_train=600)
        calib_size = max(200, int(len(X_train) * 0.15))
        calib_size = min(calib_size, max(1, len(X_train) // 3)) if len(X_train) >= 3 else 1
        train_cut = len(X_train) - calib_size
        if train_cut <= tscv.min_train:
            raise RuntimeError("Dataset insuficiente para dejar bloque de calibración sin fuga de datos.")
        X_core, y_core = X_train.iloc[:train_cut], y_train.iloc[:train_cut]
        X_calib, y_calib = X_train.iloc[train_cut:], y_train.iloc[train_cut:]
        logger.info("Split entrenamiento-core=%d | calibración=%d", len(X_core), len(X_calib))
        X_calib_cal, y_calib_cal = X_calib, y_calib
        X_thr_block, y_thr_block = X_calib, y_calib
        if len(X_calib) >= (MIN_CALIBRATION_SIZE + MIN_SIGNALS_EV):
            thr_size = max(MIN_SIGNALS_EV + 10, int(len(X_calib) * THR_HOLDOUT_FRAC))
            max_thr_size = len(X_calib) - MIN_CALIBRATION_SIZE
            thr_size = min(thr_size, max_thr_size)
            cut_idx = len(X_calib) - thr_size
            if cut_idx > 0:
                X_calib_cal, y_calib_cal = X_calib.iloc[:cut_idx], y_calib.iloc[:cut_idx]
                X_thr_block, y_thr_block = X_calib.iloc[cut_idx:], y_calib.iloc[cut_idx:]
                logger.info("Calibración hold-out dividida → cal=%d | threshold=%d", len(X_calib_cal), len(X_thr_block))
            else:
                logger.warning("No se pudo reservar bloque exclusivo para threshold (cut_idx=%d).", cut_idx)
        else:
            logger.warning("Bloque de calibración corto (%d); se reutilizará para threshold EV.", len(X_calib))

        # 7) Optuna (PR-AUC, con pruning + timeout)
        logger.info("=== Búsqueda de hiperparámetros (Optuna, PR-AUC) ===")
        train_sample_size = int(len(X_core) * SAMPLE_FRAC)
        train_sample_size = max(tscv.min_train, train_sample_size)
        X_train_sample, y_train_sample = X_core.iloc[:train_sample_size], y_core.iloc[:train_sample_size]
        best_params = train_models_optuna_pr_auc(
            X_train_sample, y_train_sample, tscv,
            n_trials=N_TRIALS, n_jobs=N_JOBS_OPTUNA, timeout=OPTUNA_TIMEOUT
        )

        # 8) Entrenamiento final
        logger.info("=== Entrenando Pipeline Final (sin bloque de calibración) ===")
        if os.getenv("USE_SMOTE", "0") == "1":
            X_fit, y_fit = balance_classes_smote(X_core, y_core)
        else:
            X_fit, y_fit = X_core, y_core
        final_pipeline = train_final_pipeline_with_params(
            X_fit, y_fit,
            best_params,
            use_pca=USE_PCA, pca_n_components=PCA_N_COMPONENTS,
            factor_pos_weight=1.0
        )

        # 9) Calibración + selección de umbral por EV (hold-out)
        cal_method = os.getenv("CAL_METHOD", "isotonic").strip().lower()
        assert cal_method in ("isotonic", "sigmoid")
        logger.info("=== Calibración %s (hold-out) ===", cal_method)
        y_prob_cal_raw = final_pipeline.predict_proba(X_calib_cal)[:, 1]

        calibrator = None
        try:
            if cal_method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_prob_cal_raw, y_calib_cal.values)
                y_prob_cal = calibrator.predict(y_prob_cal_raw)
            else:
                calibrator = LogisticRegression(max_iter=1000)
                calibrator.fit(y_prob_cal_raw.reshape(-1, 1), y_calib_cal.values)
                y_prob_cal = calibrator.predict_proba(y_prob_cal_raw.reshape(-1, 1))[:, 1]
            logger.info("Probabilidades de CALIB calibradas con %s.", cal_method)
        except Exception as e:
            y_prob_cal = y_prob_cal_raw
            calibrator = None
            logger.warning("No se aplicó calibración %s en CALIB: %s", cal_method, e)

        y_prob_thr_raw = final_pipeline.predict_proba(X_thr_block)[:, 1]
        if calibrator is not None:
            try:
                if cal_method == "isotonic":
                    y_prob_thr = calibrator.predict(y_prob_thr_raw)
                else:
                    y_prob_thr = calibrator.predict_proba(y_prob_thr_raw.reshape(-1, 1))[:, 1]
            except Exception as cal_thr_exc:
                logger.warning("No se pudo aplicar calibrador en bloque de threshold: %s", cal_thr_exc)
                y_prob_thr = y_prob_thr_raw
        else:
            y_prob_thr = y_prob_thr_raw
        y_prob_thr = postprocess_probs(y_prob_thr)

        # Subset con columnas necesarias para EV
        df_thr_full = add_trade_indicators(df_train.loc[X_thr_block.index])
        df_thr_full = df_thr_full.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        mask_long = entry_mask(df_thr_full, side="long")
        if mask_long.sum() == 0:
            logger.warning("Entry mask vacío en calibración; se usará threshold fallback.")
        p_series = pd.Series(y_prob_thr, index=df_thr_full.index).loc[mask_long]

        if p_series.empty:
            BEST_THRESHOLD = float(min(max(PROB_THRESHOLD_FALLBACK, THR_BASE_DEFAULT), THR_MAX_CAP))
            calib_signals = 0
        else:
            y_ev = y_thr_block.reindex(p_series.index)
            valid_mask = (~p_series.isna()) & (~y_ev.isna())
            if valid_mask.sum() >= MIN_SIGNALS_EV:
                bars_per_day = infer_bars_per_day_from_index(p_series.index)
                BEST_THRESHOLD, thr_stats = select_threshold_by_ev_unificado(
                    p_series[valid_mask].values,
                    y_ev[valid_mask].astype(int).values,
                    bars_per_day=bars_per_day,
                    target_trades_per_day=MIN_TRADES_DAILY_TARGET,
                    idx_times=p_series.index[valid_mask],
                )
                BEST_THRESHOLD = float(np.clip(BEST_THRESHOLD, THR_BASE_DEFAULT, THR_MAX_CAP))
                calib_signals = int((p_series >= BEST_THRESHOLD).sum())
                logger.info(
                    "Selector EV unificado → thr=%.3f | EV_mean=%.5f | EV_total=%.4f | señales=%d | trades/día=%.3f | RR=%.3f | cost_R=%.5f",
                    BEST_THRESHOLD,
                    thr_stats.get("ev_mean", 0.0),
                    thr_stats.get("ev_total", 0.0),
                    calib_signals,
                    thr_stats.get("trades_per_day", 0.0),
                    thr_stats.get("RR", 0.0),
                    thr_stats.get("cost_R", 0.0),
                )
                cand_table = thr_stats.get("candidates", [])
                if cand_table:
                    try:
                        cand_df = pd.DataFrame(cand_table)
                        cand_df = cand_df.drop_duplicates()
                        logger.info("\n[Threshold candidates]\n%s", cand_df.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
                        try:
                            tbl_path = os.path.join(MODEL_DIR, "threshold_candidates.csv")
                            cand_df.to_csv(tbl_path, index=False)
                            logger.info("[Threshold candidates] Guardado en %s", tbl_path)
                        except Exception as save_exc:
                            logger.warning("No se pudo guardar threshold_candidates.csv: %s", save_exc)
                    except Exception as cand_exc:
                        logger.debug("No se pudo mostrar tabla de thresholds: %s", cand_exc)
            else:
                BEST_THRESHOLD = float(min(max(PROB_THRESHOLD_FALLBACK, THR_BASE_DEFAULT), THR_MAX_CAP))
                calib_signals = int((p_series >= BEST_THRESHOLD).sum())
                logger.info(
                    "Selector EV unificado → fallback por falta de señales (valid=%d < MIN_SIGNALS_EV=%d)",
                    int(valid_mask.sum()),
                    MIN_SIGNALS_EV,
                )
        logger.info("Umbral final aplicado: %.3f | señales=%d", BEST_THRESHOLD, calib_signals)

        # 10) Guardar artefactos
        save_model_and_results(
            final_pipeline, best_params, feature_cols,
            model_name=MODEL_NAME, results_path=RESULTS_PATH,
            alternative_save_path=os.path.join(BASE_PATH, "backup_results"),
            best_threshold=BEST_THRESHOLD, calibrator=calibrator
        )

        # 11) Evaluación en TEST
        logger.info("=== Evaluación en Test ===")
        y_prob_test = final_pipeline.predict_proba(X_test)[:, 1]
        if calibrator is not None:
            try:
                if cal_method == "isotonic":
                    y_prob_test = calibrator.predict(y_prob_test)
                else:
                    y_prob_test = calibrator.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]
                logger.info("Probabilidades de TEST calibradas con %s.", cal_method)
            except Exception as e:
                logger.warning("No se pudo aplicar calibración %s en TEST: %s", cal_method, e)
        y_prob_test = postprocess_probs(y_prob_test)

        test_metrics, eval_artifacts = evaluate_and_plots(
            final_pipeline, X_test, y_test, y_prob_test, BEST_THRESHOLD,
            results_path=RESULTS_PATH, title_suffix="_test"
        )
        portfolio_artifacts.extend(eval_artifacts)

        # Diagnóstico EV por decil
        try:
            df_eval = pd.DataFrame(
                {
                    "y": y_test.astype(int).values,
                    "p": y_prob_test,
                }
            )
            df_eval = df_eval.dropna()
            if not df_eval.empty:
                df_eval["dec"] = pd.qcut(df_eval["p"], 10, labels=False, duplicates="drop")
                tp_mult_env = float(os.getenv("TP_MULT", 1.15))
                sl_mult_env = max(1e-8, float(os.getenv("SL_MULT", 0.60)))
                rr_env = max(0.1, tp_mult_env / sl_mult_env)
                cost_env = (
                    float(os.getenv("COMMISSION_RATE", 0.00030)) * 2
                    + float(os.getenv("SPREAD_PCT", 0.00005))
                    + float(os.getenv("SLIPPAGE_BASE", 0.00010))
                )
                tab = df_eval.groupby("dec").agg(
                    n=("y", "size"),
                    p_mean=("p", "mean"),
                    hit=("y", "mean"),
                )
                tab["ev_proxy"] = tab["hit"] * rr_env - (1 - tab["hit"]) - cost_env
                logger.info("\n[EV por decil de p_up]\n%s", tab.to_string(float_format=lambda v: f"{v:0.4f}"))
        except Exception as diag_exc:
            logger.warning("No se pudo calcular EV por deciles: %s", diag_exc)

        # 12) Guardar probabilidades y señales (para consumo externo/backtest global)
        pred_df = pd.DataFrame(
            {
                "open_time": df_test.index,
                "prob": y_prob_test,
            }
        )
        probs_path = os.path.join(RESULTS_PATH, "pred_probs.csv")
        pred_df.to_csv(probs_path, index=False)
        portfolio_artifacts.append(probs_path)
        raw_signals = (y_prob_test >= BEST_THRESHOLD).astype(int)
        signals_path = os.path.join(RESULTS_PATH, "signals_binary.csv")
        pd.DataFrame({'open_time': df_test.index, 'raw_signal': raw_signals})\
          .to_csv(signals_path, index=False)
        portfolio_artifacts.append(signals_path)

        if GENERATE_FEATURE_IMPORTANCE:
            fi_artifacts = compute_feature_importance_artifacts(
                final_pipeline, X_test, y_test, RESULTS_PATH, suffix="_test"
            )
            portfolio_artifacts.extend(fi_artifacts)

        if GENERATE_PORTFOLIO_REPORT:
            optuna_cfg = {
                "sample_frac": SAMPLE_FRAC,
                "n_trials": N_TRIALS,
                "timeout_s": OPTUNA_TIMEOUT,
                "n_jobs_optuna": N_JOBS_OPTUNA,
                "cv_splits": getattr(tscv, "n_splits", 5),
                "embargo": FIXED_LOOK_AHEAD,
            }
            report_path = generate_training_report(
                RESULTS_PATH,
                MODEL_NAME,
                dataset_summary,
                best_params,
                BEST_THRESHOLD,
                test_metrics,
                optuna_cfg,
                cal_method,
                portfolio_artifacts,
            )
            if report_path:
                portfolio_artifacts.append(report_path)

        logger.info("Señales y probabilidades guardadas.")
        logger.info("=== Proceso Completado con Éxito ===")
    except Exception as e:
        logger.error(f"Error en main(): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
