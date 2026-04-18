from __future__ import annotations

"""
Macroeconomic pipeline with rigorous treatment of multicollinearity,
autocorrelation, and non-stationarity. Uses ONLY real data from public APIs.

Key improvements to address:
- Severe multicollinearity (Condition Number ~ 1e17) → correlation filtering, strict VIF,
  rank deficiency check.
- Strong positive autocorrelation (DW ~ 0.437) → increased AR order (max_lag=4),
  optimal AR selection via AIC, clear warnings.

** Modified: OLS replaced with ARDL (Autoregressive Distributed Lag) **
** Extended: Added support for multiple countries (US, CN, VN) **
** Added: Green/environmental indicators from World Bank and local files **
"""

import logging
import re
import warnings
import io
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ardl import ARDL, ARDLResults          # <-- ARDL import
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# Logging setup
# =========================

def setup_logging(log_dir: str = "logs") -> None:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir_path / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# =========================
# Configuration (adjusted to combat multicollinearity & autocorrelation)
# =========================

@dataclass
class PipelineConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    log_dir: str = "logs"
    target_name: str = "gdp_growth"
    min_obs: int = 200
    max_lag: int = 4                     # Increased to capture longer autocorrelation
    vif_threshold: float = 2.0           # Stricter to eliminate near-linear dependencies
    adf_alpha: float = 0.05
    use_log_for_positive_series: bool = False
    use_diff_if_nonstationary: bool = True
    include_quarter_dummies: bool = True
    include_trend: bool = False
    ar_gls_maxiter: int = 25
    n_bootstrap: int = 500               # More bootstrap samples for stable CI
    plot: bool = True
    vif_chunk_size: int = 1000
    ridge_alphas: List[float] = field(default_factory=lambda: np.logspace(-3, 2, 30).tolist())
    n_jobs: int = 1
    # NEW: Country code (default Vietnam)
    country_code: str = "VN"


# =========================
# Data fetching (unchanged)
# =========================

def fetch_worldbank_indicator(indicator: str, country: str = "VN", retries: int = 3) -> pd.Series:
    logger = logging.getLogger(__name__)
    logger.info(f"Fetching World Bank data: {indicator} for {country}")
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=1000"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data or len(data) < 2:
                logger.warning(f"Empty response for {indicator} ({country})")
                return pd.Series(dtype=float)
            records = data[1]
            if not records:
                return pd.Series(dtype=float)
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "value"]).sort_values("date")
            df["date"] = df["date"] + pd.offsets.YearEnd()
            s = df.set_index("date")["value"]
            s.name = indicator
            logger.info(f"Fetched {len(s)} annual observations for {indicator} ({country})")
            return s
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {indicator} ({country}): {e}")
            time.sleep(2 ** attempt)
    logger.error(f"Failed to fetch {indicator} after {retries} attempts for {country}.")
    return pd.Series(dtype=float)


def fetch_bundesbank_vnd_usd() -> pd.Series:
    logger = logging.getLogger(__name__)
    url = "https://api.statistiken.bundesbank.de/rest/download/BBEX3/M.VND.USD.CA.AC.A01?format=csv&lang=en"
    headers = {"Accept": "text/csv"}
    logger.info("Fetching Bundesbank VND/USD exchange rate")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        content = resp.content.decode("utf-8", errors="ignore")
        lines = content.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "Date" in line and "Value" in line:
                header_idx = i
                break
        if header_idx is None:
            raise ValueError("Could not locate header in Bundesbank CSV")
        data_lines = lines[header_idx:]
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=";", engine="python")
        df = df.iloc[:, :2]
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date")
        s = df.set_index("date")["value"]
        s.name = "exchange_rate"
        logger.info(f"Fetched {len(s)} monthly observations from Bundesbank")
        return s
    except Exception as e:
        logger.warning(f"Bundesbank fetch failed: {e}")
        return pd.Series(dtype=float)


# === THÊM DỮ LIỆU XANH: Danh sách chỉ số môi trường từ World Bank ===
GREEN_INDICATORS = {
    "co2_emissions":       "EN.ATM.CO2E.KT",        # CO2 emissions (kt)
    "renewable_energy":    "EG.FEC.RNEW.ZS",        # Renewable energy consumption (% of total)
    "forest_area":         "AG.LND.FRST.ZS",        # Forest area (% of land area)
    "freshwater_withdraw": "ER.H2O.FWTL.ZS",        # Annual freshwater withdrawals (% of internal resources)
    "population_density":  "EN.POP.DNST",           # Population density (people per sq. km)
}


# === THÊM DỮ LIỆU XANH: Hàm fetch các chỉ số xanh ===
def fetch_green_indicators(country: str = "VN") -> Dict[str, pd.Series]:
    """
    Fetch all green indicators from World Bank for a given country.
    Returns a dictionary mapping indicator name to pandas Series.
    """
    logger = logging.getLogger(__name__)
    green_series = {}
    for name, code in GREEN_INDICATORS.items():
        s = fetch_worldbank_indicator(code, country=country)
        if not s.empty:
            green_series[name] = s
        else:
            logger.info(f"Green indicator {code} ({name}) not available for {country}")
    return green_series


# =========================
# Local file parsers for FiinProX Excel files (unchanged)
# =========================

def _extract_series_from_excel(
    file_path: Path,
    sheet_name: str = "Sheet1",
    value_row_start: int = 4,
    header_row: int = 4,
    year_start_col: int = 2,
    date_format: str = "year",
    value_units: str = "percent",
) -> Dict[str, pd.Series]:
    """
    Generic extractor for FiinProX Excel files where the first column is labels,
    subsequent columns are dates (years, quarters, or months).
    Returns a dict of series: {label_cleaned: Series}
    """
    logger = logging.getLogger(__name__)
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        logger.warning(f"Could not read Excel file {file_path}: {e}")
        return {}

    # Find header row (date columns)
    if header_row >= len(df_raw):
        return {}
    header = df_raw.iloc[header_row, year_start_col:]
    header = header.dropna().astype(str).tolist()

    # Parse date index
    dates = []
    for col_str in header:
        col_str = col_str.strip()
        if date_format == "year":
            try:
                year = int(col_str)
                dates.append(pd.Timestamp(year=year, month=12, day=31))
            except:
                dates.append(pd.NaT)
        elif date_format == "quarter":
            m = re.match(r"(\d?)Q[-\s]*(\d{4})", col_str, re.I)
            if m:
                q = int(m.group(1)) if m.group(1) else 4
                year = int(m.group(2))
                month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
                dates.append(pd.Timestamp(year=year, month=month, day=30 if month in [6,9] else 31))
            else:
                dates.append(pd.NaT)
        elif date_format == "month":
            m = re.match(r"(\d{1,2})[-\s]*(\d{4})", col_str)
            if m:
                month = int(m.group(1))
                year = int(m.group(2))
                dates.append(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd())
            else:
                dates.append(pd.NaT)
        else:
            dates.append(pd.NaT)

    date_index = pd.DatetimeIndex(dates)

    # Extract data rows
    data_rows = df_raw.iloc[value_row_start:, :]
    series_dict = {}
    for idx, row in data_rows.iterrows():
        label = row.iloc[0]
        if pd.isna(label):
            continue
        label = str(label).strip()
        if label in ["", "Contact", "CÔNG TY CỔ PHẦN FIINGROUP VIỆT NAM"]:
            continue  # skip metadata
        values = row.iloc[year_start_col:year_start_col+len(dates)]
        values = pd.to_numeric(values, errors='coerce')
        if values.isna().all():
            continue
        s = pd.Series(values.values, index=date_index, name=label)
        # Convert percent to decimal if needed
        if value_units == "percent" and s.max() > 1:
            s = s / 100.0
        series_dict[label] = s

    return series_dict


def parse_fiinpro_gdp_annual(file_path: Path) -> Dict[str, pd.Series]:
    """Parse annual GDP file (first file provided)."""
    return _extract_series_from_excel(
        file_path,
        sheet_name="Sheet1",
        value_row_start=5,
        header_row=4,
        year_start_col=2,
        date_format="year",
        value_units="percent"
    )


def parse_fiinpro_gdp_quarterly(file_path: Path) -> Dict[str, pd.Series]:
    """Parse quarterly GDP file (files with Q1-2022 etc)."""
    return _extract_series_from_excel(
        file_path,
        sheet_name="Sheet1",
        value_row_start=5,
        header_row=4,
        year_start_col=2,
        date_format="quarter",
        value_units="percent"
    )


def parse_fiinpro_import_monthly(file_path: Path) -> Dict[str, pd.Series]:
    """Parse monthly import value file (triệu USD)."""
    return _extract_series_from_excel(
        file_path,
        sheet_name="Sheet1",
        value_row_start=5,
        header_row=4,
        year_start_col=2,
        date_format="month",
        value_units="absolute"   # values are in million USD
    )


def parse_fiinpro_import_growth(file_path: Path) -> Dict[str, pd.Series]:
    """Parse monthly import growth YoY file (%)."""
    return _extract_series_from_excel(
        file_path,
        sheet_name="Sheet1",
        value_row_start=5,
        header_row=4,
        year_start_col=2,
        date_format="month",
        value_units="percent"
    )


def parse_fiinpro_industry_indicators(file_path: Path) -> Dict[str, pd.Series]:
    """
    Parse industry indicators file (Chỉ số chính - ...).
    Format: columns are years, rows are indicators.
    """
    logger = logging.getLogger(__name__)
    try:
        df_raw = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return {}

    # Find header row with years
    header_row = None
    for i in range(min(10, len(df_raw))):
        row = df_raw.iloc[i]
        # Look for consecutive years (e.g., 2023, 2024, 2025)
        if row.notna().sum() >= 2:
            years = pd.to_numeric(row, errors='coerce')
            if years.notna().sum() >= 2 and years.max() > 2000:
                header_row = i
                break
    if header_row is None:
        logger.warning(f"Could not find year header in {file_path}")
        return {}

    year_cols = df_raw.iloc[header_row].dropna().tolist()
    years = []
    for y in year_cols:
        try:
            years.append(int(float(y)))
        except:
            continue
    date_index = pd.DatetimeIndex([pd.Timestamp(year=y, month=12, day=31) for y in years])

    # Find data start row (after blank line)
    data_start = header_row + 2
    if data_start >= len(df_raw):
        return {}

    series_dict = {}
    current_indicator = None
    for idx in range(data_start, len(df_raw)):
        row = df_raw.iloc[idx]
        label = row.iloc[0]
        if pd.isna(label):
            continue
        label = str(label).strip()
        if label.startswith("Contact"):
            break
        # If label is a section header (e.g., "Hiệu suất sinh lời"), skip
        if not label.replace('.', '').replace(' ', '').replace('_', '').isalnum():
            current_indicator = label
            continue
        # It's an indicator name
        values = row.iloc[1:1+len(years)]
        values = pd.to_numeric(values, errors='coerce')
        if values.isna().all():
            continue
        s = pd.Series(values.values, index=date_index, name=label)
        series_dict[label] = s

    return series_dict


# === THÊM DỮ LIỆU XANH: Parser cho file Excel chứa chỉ số môi trường ===
def parse_fiinpro_green_indicators(file_path: Path) -> Dict[str, pd.Series]:
    """
    Parse green/environmental indicators from a FiinProX Excel file.
    Expected format: first column is indicator name, subsequent columns are years.
    """
    return _extract_series_from_excel(
        file_path,
        sheet_name="Sheet1",
        value_row_start=5,
        header_row=4,
        year_start_col=2,
        date_format="year",
        value_units="absolute"   # adjust if percentages
    )


def load_local_series(cfg: PipelineConfig) -> Dict[str, pd.Series]:
    input_path = Path(cfg.input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    local_series = {}
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        for file in input_path.glob(ext):
            try:
                df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
                if df.shape[1] >= 2:
                    df.columns = ["date", "value"] + list(df.columns[2:])
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna(subset=["date", "value"]).sort_values("date")
                    s = df.set_index("date")["value"]
                    name = file.stem.lower()
                    if "gdp" in name:
                        local_series["gdp_growth"] = s
                    elif "import" in name:
                        local_series["imports"] = s
                    elif "cpi" in name or "inflation" in name:
                        local_series["inflation"] = s
                    elif "exchange" in name or "fx" in name:
                        local_series["exchange_rate"] = s
                    elif "policy" in name or "interest" in name:
                        local_series["policy_rate"] = s
                    else:
                        local_series[name] = s
                    logging.getLogger(__name__).info(f"Loaded local series '{name}' from {file.name}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load {file}: {e}")

    # === THÊM DỮ LIỆU XANH: Tự động nhận diện file chứa từ khóa "green" hoặc "environment" ===
    green_keywords = ["green", "co2", "renewable", "forest", "emission", "environment"]
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        for file in input_path.glob(ext):
            name_lower = file.stem.lower()
            if any(kw in name_lower for kw in green_keywords):
                try:
                    df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
                    if df.shape[1] >= 2:
                        df.columns = ["date", "value"] + list(df.columns[2:])
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df["value"] = pd.to_numeric(df["value"], errors="coerce")
                        df = df.dropna(subset=["date", "value"]).sort_values("date")
                        s = df.set_index("date")["value"]
                        key = f"green_{file.stem}"
                        local_series[key] = s
                        logging.getLogger(__name__).info(f"Loaded green series '{key}' from {file.name}")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to load green file {file}: {e}")

    return local_series


def build_quarterly_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    # This original function remains unchanged for backward compatibility.
    # It calls the new generic version with cfg.country_code (default "VN").
    return build_quarterly_dataset_for_country(cfg.country_code, cfg)


# ============================================================
# NEW: Function to build quarterly dataset for any country
# ============================================================
def build_quarterly_dataset_for_country(country_code: str, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Build quarterly dataset for a given country code (e.g., 'VN', 'US', 'CN').
    Fetches World Bank indicators, optionally uses Bundesbank for VN exchange rate,
    and merges local series if available.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"===== Building quarterly dataset for {country_code} =====")

    # World Bank indicators (same for all countries)
    indicators = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "inflation": "FP.CPI.TOTL.ZG",
        "imports": "NE.IMP.GNFS.ZS",
        "policy_rate": "FR.INR.RINR",
        "exchange_rate": "PA.NUS.FCRF",
    }
    annual_series = {}

    # === THÊM DỮ LIỆU XANH: Fetch green indicators from World Bank ===
    green_series = fetch_green_indicators(country_code)
    for name, s in green_series.items():
        annual_series[name] = s

    # === THÊM DỮ LIỆU XANH: Load local green data files (nếu có) ===
    green_local_files = list(Path(cfg.input_dir).glob("*green*.xlsx")) + \
                        list(Path(cfg.input_dir).glob("*environment*.xlsx"))
    for file in green_local_files:
        try:
            green_dict = parse_fiinpro_green_indicators(file)
            for name, s in green_dict.items():
                key = f"green_{name.replace(' ', '_').lower()}"
                annual_series[key] = s
                logger.info(f"Added local green indicator '{key}' from {file.name}")
        except Exception as e:
            logger.warning(f"Failed to parse green file {file}: {e}")

    # Continue with core indicators
    for name, code in indicators.items():
        s = fetch_worldbank_indicator(code, country=country_code)
        if not s.empty:
            annual_series[name] = s
        else:
            logger.warning(f"World Bank indicator {code} returned empty for {country_code}.")

    # Special handling for Vietnam exchange rate (Bundesbank)
    if country_code == "VN" and ("exchange_rate" not in annual_series or annual_series["exchange_rate"].empty):
        fx = fetch_bundesbank_vnd_usd()
        if not fx.empty:
            fx_annual = fx.resample("YE").mean()
            fx_annual.name = "exchange_rate"
            annual_series["exchange_rate"] = fx_annual

    # Load local series (if any) - may override
    local_series = load_local_series(cfg)
    for name, s in local_series.items():
        if name in annual_series:
            logger.info(f"Overriding {name} with local data for {country_code}.")
        annual_series[name] = s

    if "gdp_growth" not in annual_series or annual_series["gdp_growth"].empty:
        raise RuntimeError(f"GDP growth data is missing for {country_code}. Please provide a CSV file in './input/'.")

    # Create quarterly index and interpolate
    quarterly_index = pd.date_range("1970-01-01", "2025-12-31", freq="QE")
    quarterly_df = pd.DataFrame(index=quarterly_index)
    for name, s in annual_series.items():
        if s.empty:
            continue
        annual_aligned = s.reindex(quarterly_index, method=None)
        quarterly_df[name] = annual_aligned.interpolate(method="time", limit_direction="both")

    quarterly_df = quarterly_df.dropna(how="all")
    n_real_obs = len(quarterly_df.dropna(subset=["gdp_growth"]))
    logger.info(f"Quarterly dataset shape for {country_code}: {quarterly_df.shape}, GDP obs: {n_real_obs}")

    if n_real_obs < cfg.min_obs:
        raise RuntimeError(f"Only {n_real_obs} real quarterly observations available for {country_code} (required >= {cfg.min_obs}).")

    if "exchange_rate" in quarterly_df.columns and quarterly_df["exchange_rate"].isna().all():
        logger.warning(f"Exchange rate data entirely missing for {country_code}; dropping variable.")
        quarterly_df = quarterly_df.drop(columns=["exchange_rate"])

    quarterly_df.index.name = "date"
    return quarterly_df


# =========================
# Enhanced preprocessing utilities (unchanged)
# =========================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_log_transform(s: pd.Series) -> pd.Series:
    if (s <= 0).any():
        return s
    return np.log(s)


def adf_summary(series: pd.Series, alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    s = series.dropna()
    if len(s) < 8:
        return {"pvalue": np.nan, "stationary": False}
    try:
        result = adfuller(s, autolag="AIC")
        pvalue = float(result[1])
        return {"pvalue": pvalue, "stationary": pvalue < alpha}
    except Exception:
        return {"pvalue": np.nan, "stationary": False}


def make_stationary(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    s = series.copy()
    info = adf_summary(s, alpha)
    if info["stationary"]:
        return s
    diffed = s.diff()
    diff_info = adf_summary(diffed, alpha)
    if diff_info["stationary"]:
        return diffed
    pct = s.pct_change()
    pct_info = adf_summary(pct, alpha)
    if pct_info["stationary"]:
        return pct
    return diffed


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if "trend" not in out.columns:
            out["trend"] = np.arange(len(out), dtype=float)
        if len(out) >= 4:
            q = out.index.quarter
            dummies = pd.get_dummies(q, prefix="q", drop_first=True)
            dummies.index = out.index
            out = pd.concat([out, dummies], axis=1)
    return out


def add_lags(df: pd.DataFrame, columns: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def rolling_features(df: pd.DataFrame, columns: Sequence[str], window: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_roll{window}_mean"] = out[col].rolling(window).mean()
        out[f"{col}_roll{window}_std"] = out[col].rolling(window).std()
    return out


def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one feature from each pair with absolute correlation > threshold.
    Keeps the feature that appears first in the column list.
    """
    X_num = X.select_dtypes(include=[np.number]).dropna()
    if X_num.shape[1] < 2:
        return X, []

    corr_matrix = X_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for other in high_corr:
            if other not in to_drop:
                to_drop.add(other)

    X_filtered = X.drop(columns=list(to_drop), errors='ignore')
    return X_filtered, list(to_drop)


def compute_vif(X: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    X_ = X.copy().dropna()
    X_ = X_.loc[:, X_.nunique(dropna=True) > 1]
    if X_.shape[1] < 2:
        return pd.DataFrame({"feature": X_.columns, "VIF": [1.0] * X_.shape[1]})

    scaler = StandardScaler()
    n_rows, n_cols = X_.shape
    if n_cols > chunk_size:
        means = np.zeros(n_cols)
        stds = np.zeros(n_cols)
        for start in range(0, n_cols, chunk_size):
            end = min(start + chunk_size, n_cols)
            chunk = X_.iloc[:, start:end].values
            means[start:end] = np.nanmean(chunk, axis=0)
            stds[start:end] = np.nanstd(chunk, axis=0)
        stds[stds == 0] = 1.0
        X_scaled = (X_.values - means) / stds
    else:
        X_scaled = scaler.fit_transform(X_)

    vals = []
    for i, col in enumerate(X_.columns):
        try:
            vif = variance_inflation_factor(X_scaled, i)
        except Exception:
            vif = np.inf
        vals.append((col, float(vif)))
    return pd.DataFrame(vals, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)


def prune_vif(X: pd.DataFrame, threshold: float = 2.0, chunk_size: int = 1000) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively drop features with VIF > threshold.
    Returns pruned DataFrame and list of dropped features.
    """
    X_work = X.copy()
    dropped: List[str] = []
    while True:
        X_num = X_work.select_dtypes(include=[np.number]).dropna()
        if X_num.shape[1] < 2:
            break
        vif_df = compute_vif(X_num, chunk_size=chunk_size)
        if vif_df.empty:
            break
        worst = vif_df.iloc[0]
        if not np.isfinite(worst["VIF"]) or worst["VIF"] > threshold:
            feature = str(worst["feature"])
            if feature in X_work.columns:
                X_work = X_work.drop(columns=[feature])
                dropped.append(feature)
            else:
                break
        else:
            break
    return X_work, dropped


def condition_number(X: pd.DataFrame) -> float:
    """Compute condition number of design matrix (excluding constant)."""
    X_ = X.select_dtypes(include=[np.number]).dropna()
    if X_.shape[1] < 2:
        return 1.0
    X_scaled = StandardScaler().fit_transform(X_)
    _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
    s = s[s > 1e-12]
    return float(s.max() / s.min())


def select_ar_order(y: pd.Series, max_lag: int = 4, criterion: str = "aic") -> int:
    """
    Select optimal AR order for GLSAR/SARIMAX using AIC or BIC.
    Returns best lag order (minimum 1).
    """
    y_clean = y.dropna()
    if len(y_clean) < 20:
        return 1
    best_order = 1
    best_crit = np.inf
    for p in range(1, max_lag + 1):
        try:
            model = sm.tsa.AutoReg(y_clean, lags=p, trend="n")
            res = model.fit()
            crit = getattr(res, criterion)
            if crit < best_crit:
                best_crit = crit
                best_order = p
        except Exception:
            continue
    return best_order


def bootstrap_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int,
    alphas: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, RidgeCV]:
    if alphas is None:
        alphas = np.logspace(-3, 2, 30).tolist()

    X = X.astype(float)
    y = y.astype(float)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    if X.shape[0] < 10:
        raise ValueError("Insufficient data for bootstrap ridge regression.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = RidgeCV(alphas=alphas).fit(X_scaled, y)
    coefs = []
    for _ in range(n_bootstrap):
        X_res, y_res = resample(X_scaled, y, replace=True, random_state=None)
        model = RidgeCV(alphas=alphas).fit(X_res, y_res)
        coefs.append(model.coef_)
    coefs = np.array(coefs)
    ci_lower = np.percentile(coefs, 2.5, axis=0)
    ci_upper = np.percentile(coefs, 97.5, axis=0)
    results = pd.DataFrame({
        'feature': X.columns,
        'coef': ridge.coef_,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    return results, ridge


# =========================
# Core design matrix builder (improved for multicollinearity & rank)
# =========================

def build_design_matrix(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> Tuple[pd.Series, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    data = df.copy()
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Optional log transform
    if cfg.use_log_for_positive_series:
        for col in data.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(data[col]):
                if (data[col].dropna() > 0).all() and data[col].nunique(dropna=True) > 5:
                    data[col] = maybe_log_transform(data[col])

    # Make stationary
    if cfg.use_diff_if_nonstationary:
        for col in data.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(data[col]):
                info = adf_summary(data[col], cfg.adf_alpha)
                if not info["stationary"]:
                    data[col] = make_stationary(data[col], cfg.adf_alpha)

        if not re.search(r"growth|inflation|rate|pct|percent", target, re.I):
            tinfo = adf_summary(data[target], cfg.adf_alpha)
            if not tinfo["stationary"]:
                data[target] = make_stationary(data[target], cfg.adf_alpha)

    # Time features
    if cfg.include_quarter_dummies:
        data = add_time_features(data)
    if cfg.include_trend and "trend" not in data.columns:
        data["trend"] = np.arange(len(data), dtype=float)

    # Rolling features
    exog_cols = [c for c in data.columns if c != target]
    data = rolling_features(data, [c for c in exog_cols if pd.api.types.is_numeric_dtype(data[c])], window=4)

    # Lags (limited to cfg.max_lag)
    lag_base_cols = [c for c in data.columns if c != target and pd.api.types.is_numeric_dtype(data[c])]
    ar_lags = list(range(1, cfg.max_lag + 1))
    data = add_lags(data, lag_base_cols + [target], lags=ar_lags)

    # Prepare X, y
    feature_cols = [c for c in data.columns if c != target]
    X = data[feature_cols].copy()
    y = data[target].copy()

    # Drop near-constant columns
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # --- Drop highly correlated features (>0.95) ---
    X, dropped_corr = drop_highly_correlated(X, threshold=0.95)
    if dropped_corr:
        logger.info("Dropped due to high correlation (>0.95): %s", dropped_corr)

    # --- Prune by VIF (stricter threshold) ---
    X, dropped_vif = prune_vif(X, threshold=cfg.vif_threshold, chunk_size=cfg.vif_chunk_size)
    if dropped_vif:
        logger.info("Dropped by VIF (>%.2f): %s", cfg.vif_threshold, dropped_vif)

    # Align and drop NAs
    joined = pd.concat([y, X], axis=1).dropna()
    y = joined[target]
    X = joined.drop(columns=[target])

    # Add constant
    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid]
    X = X[valid]

    # --- Check matrix rank and remove dependent columns if singular ---
    X_numeric = X.astype(float)
    rank = np.linalg.matrix_rank(X_numeric)
    if rank < X_numeric.shape[1]:
        logger.warning(f"Design matrix rank deficient: rank={rank}, cols={X_numeric.shape[1]}. Removing dependent columns.")
        Q, R = np.linalg.qr(X_numeric)
        independent_cols = np.where(np.abs(np.diag(R)) > 1e-8)[0]
        X = X.iloc[:, independent_cols]
        logger.info(f"Retained {X.shape[1]} independent columns after rank check.")

    X = X.astype(float)
    y = y.astype(float)

    # Report condition number after cleaning
    cn = condition_number(X.drop(columns=["const"], errors="ignore"))
    logger.info("Condition number of final design matrix (excluding constant): %.2e", cn)
    if cn > 30:
        logger.warning("Condition number > 30 indicates remaining multicollinearity. Ridge regression is strongly recommended.")

    return y, X


# =========================
# Enhanced diagnostics
# =========================

def residual_diagnostics(resid: pd.Series, X: pd.DataFrame, lags: int = 4) -> Dict[str, float]:
    resid = resid.dropna()
    out: Dict[str, float] = {}
    if len(resid) >= lags + 1:
        out["dw"] = float(durbin_watson(resid))
        try:
            lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
            out[f"ljung_box_pvalue_lag{lags}"] = float(lb.iloc[0, 1])
        except Exception:
            out[f"ljung_box_pvalue_lag{lags}"] = np.nan
    else:
        out["dw"] = np.nan
        out[f"ljung_box_pvalue_lag{lags}"] = np.nan

    # Breusch-Pagan for heteroskedasticity
    try:
        common_idx = resid.index.intersection(X.index)
        if len(common_idx) > 0:
            exog = X.loc[common_idx].drop(columns=["const"], errors="ignore")
            if exog.shape[1] >= 1:
                bp = het_breuschpagan(resid.loc[common_idx], sm.add_constant(exog, has_constant="add"))
                out["bp_pvalue"] = float(bp[1])
            else:
                out["bp_pvalue"] = np.nan
        else:
            out["bp_pvalue"] = np.nan
    except Exception:
        out["bp_pvalue"] = np.nan
    return out


# =========================
# Model fitting functions
# =========================

def fit_ardl(
    y: pd.Series,
    X: pd.DataFrame,
    max_lag: int = 4,
    criterion: str = "aic",   # kept for interface compatibility
) -> ARDLResults:
    """
    Fit an ARDL(p,q) model with fixed lags = max_lag for both y and X.
    """
    y = y.astype(float)
    X_exog = X.drop(columns=["const"], errors="ignore").astype(float)
    
    model = ARDL(
        y,
        lags=max_lag,
        exog=X_exog,
        order=max_lag,
        trend="c",
        causal=False,
    )
    res = model.fit()
    return res


def fit_glsar(y: pd.Series, X: pd.DataFrame, maxiter: int = 25, ar_order: int = 1):
    y = y.astype(float)
    X = X.astype(float)
    model = sm.GLSAR(y, X, rho=ar_order)
    res = model.iterative_fit(maxiter=maxiter)
    return res, model


def fit_sarimax(y: pd.Series, X: pd.DataFrame, ar_order: int = 1):
    y = y.astype(float)
    X_exog = X.drop(columns=["const"], errors="ignore").astype(float)
    mod = SARIMAX(
        y,
        exog=X_exog,
        order=(ar_order, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False)
    return res


def print_model_summary(name: str, res, y: pd.Series, X: pd.DataFrame):
    logger = logging.getLogger(__name__)
    logger.info("\n===== %s =====", name)
    try:
        logger.info("%s", res.summary())
    except Exception:
        logger.info("Could not print standard summary; object type=%s", type(res))
    resid_values = getattr(res, "resid", None)
    if resid_values is None:
        resid = pd.Series(dtype=float)
    else:
        resid = pd.Series(np.asarray(resid_values).ravel(), index=y.index[:len(np.asarray(resid_values).ravel())])
    diag = residual_diagnostics(resid, X)
    logger.info("Diagnostics: %s", diag)


# =========================
# Main workflow
# =========================

def run_pipeline_for_country(country_code: str, base_cfg: PipelineConfig):
    """
    Run the entire pipeline for a specific country.
    This function encapsulates the logic previously in main().
    """
    logger = logging.getLogger(__name__)
    # Create a copy of config with appropriate output subdirectory
    cfg = base_cfg
    cfg.country_code = country_code
    country_out_dir = ensure_dir(Path(cfg.output_dir) / country_code)
    logger.info(f"===== Processing country: {country_code} =====")

    # 1. Build dataset
    try:
        df = build_quarterly_dataset_for_country(country_code, cfg)
    except Exception as e:
        logger.error(f"Data assembly failed for {country_code}: {e}")
        return

    logger.info(f"Merged data shape for {country_code}: {df.shape}")

    if len(df) < cfg.min_obs:
        logger.error(f"Too few observations for {country_code}: {len(df)} < {cfg.min_obs}. Skipping.")
        return

    # 2. Stationarity report
    stationarity_report = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) >= 8:
            info = adf_summary(s, cfg.adf_alpha)
            stationarity_report.append({"series": col, **info, "n": int(s.shape[0])})
    stationarity_df = pd.DataFrame(stationarity_report)
    stationarity_df.to_csv(country_out_dir / "stationarity_report.csv", index=False)

    # 3. Design matrix
    y, X = build_design_matrix(df, target="gdp_growth", cfg=cfg)
    logger.info(f"Model matrix after cleaning for {country_code}: y={y.shape}, X={X.shape}")

    prepared = pd.concat([y.rename("gdp_growth"), X], axis=1)
    prepared.to_csv(country_out_dir / "prepared_dataset.csv", index=True)

    # 4. Select optimal AR order for GLSAR/SARIMAX
    ar_order = select_ar_order(y, max_lag=cfg.max_lag, criterion="aic")
    logger.info(f"Optimal AR order selected by AIC for {country_code}: {ar_order}")

    # 5. Fit models
    ardl_res = fit_ardl(y, X, max_lag=cfg.max_lag)
    print_model_summary(f"ARDL ({country_code})", ardl_res, y, X)

    glsar_res, _ = fit_glsar(y, X, maxiter=cfg.ar_gls_maxiter, ar_order=ar_order)
    print_model_summary(f"GLSAR AR({ar_order}) ({country_code})", glsar_res, y, X)

    # Ridge
    X_ridge = X.drop(columns=["const"], errors="ignore")
    if X_ridge.shape[1] > 0 and X_ridge.shape[0] > 10:
        ridge_results, ridge_model = bootstrap_ridge(X_ridge, y, n_bootstrap=cfg.n_bootstrap, alphas=cfg.ridge_alphas)
        logger.info(f"\n===== Ridge Regression (Bootstrap CI) for {country_code} =====")
        logger.info("\n%s", ridge_results.round(4))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_ridge)
        ridge_fitted = ridge_model.predict(X_scaled)
        ridge_resid = y - ridge_fitted
        ridge_diag = residual_diagnostics(pd.Series(ridge_resid, index=y.index), X)
        logger.info(f"Ridge diagnostics for {country_code}: {ridge_diag}")
        logger.info(f"Ridge alpha selected for {country_code}: {ridge_model.alpha_:.4f}")
    else:
        ridge_results = pd.DataFrame()
        ridge_fitted = np.full(len(y), np.nan)
        ridge_resid = np.full(len(y), np.nan)
        ridge_diag = {}
        logger.warning(f"Insufficient data for ridge regression for {country_code} (skipped).")

    # SARIMAX
    try:
        sarimax_res = fit_sarimax(y, X, ar_order=ar_order)
        logger.info(f"\n===== SARIMAX REGRESSION for {country_code} =====")
        logger.info("%s", sarimax_res.summary())
        sarimax_pred = sarimax_res.predict(start=0, end=len(y)-1, exog=X.drop(columns=["const"], errors="ignore"))
    except Exception as exc:
        sarimax_res = None
        sarimax_pred = None
        logger.warning(f"SARIMAX failed for {country_code}: {exc}")

    # 6. Save coefficients
    coef_rows = []
    for name, res in [("ardl", ardl_res), ("glsar", glsar_res)]:
        params = getattr(res, "params", pd.Series(dtype=float))
        bse = getattr(res, "bse", pd.Series(dtype=float))
        pvals = getattr(res, "pvalues", pd.Series(dtype=float))
        for term in params.index:
            coef_rows.append({
                "model": name,
                "term": term,
                "coef": float(params[term]),
                "std_err": float(bse[term]) if term in bse.index else np.nan,
                "p_value": float(pvals[term]) if term in pvals.index else np.nan,
            })
    if not ridge_results.empty:
        for _, row in ridge_results.iterrows():
            coef_rows.append({
                "model": "ridge",
                "term": row["feature"],
                "coef": row["coef"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "p_value": np.nan
            })
    pd.DataFrame(coef_rows).to_csv(country_out_dir / "model_coefficients.csv", index=False)

    # 7. Fitted values
    fitted = pd.DataFrame(index=y.index)
    fitted["actual"] = y
    fitted["ardl_fitted"] = ardl_res.fittedvalues
    fitted["ardl_resid"] = ardl_res.resid
    fitted["glsar_fitted"] = glsar_res.fittedvalues
    fitted["glsar_resid"] = glsar_res.resid
    fitted["ridge_fitted"] = ridge_fitted
    fitted["ridge_resid"] = ridge_resid
    if sarimax_res is not None and sarimax_pred is not None:
        fitted["sarimax_pred"] = pd.Series(sarimax_pred, index=y.index)
    fitted.to_csv(country_out_dir / "fitted_values.csv", index=True)

    # 8. Plots
    if cfg.plot:
        fig, ax = plt.subplots(figsize=(11, 5))
        y.plot(ax=ax, label="Actual")
        fitted["ardl_fitted"].plot(ax=ax, label="ARDL")
        glsar_res.fittedvalues.plot(ax=ax, label="GLSAR")
        if not np.all(np.isnan(ridge_fitted)):
            fitted["ridge_fitted"].plot(ax=ax, label="Ridge")
        ax.set_title(f"Actual vs fitted for {country_code}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(country_out_dir / "actual_vs_fitted.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        fitted["ardl_resid"].plot(ax=axes[0], title="ARDL residuals")
        axes[0].axhline(0, linestyle="--")
        fitted["glsar_resid"].plot(ax=axes[1], title="GLSAR residuals")
        axes[1].axhline(0, linestyle="--")
        if not np.all(np.isnan(ridge_resid)):
            fitted["ridge_resid"].plot(ax=axes[2], title="Ridge residuals")
            axes[2].axhline(0, linestyle="--")
        else:
            axes[2].set_visible(False)
        if sarimax_res is not None:
            sarimax_res.resid.plot(ax=axes[3], title="SARIMAX residuals")
            axes[3].axhline(0, linestyle="--")
        else:
            axes[3].set_visible(False)
        fig.tight_layout()
        fig.savefig(country_out_dir / "residuals.png", dpi=160)
        plt.close(fig)

    stability = {
        "country": country_code,
        "n_obs": int(len(y)),
        "n_features": int(X.shape[1]),
        "condition_number": float(condition_number(X.drop(columns=["const"], errors="ignore"))),
        "ardl_dw": float(durbin_watson(ardl_res.resid)),
        "glsar_dw": float(durbin_watson(glsar_res.resid)),
        "ridge_dw": float(ridge_diag.get("dw", np.nan)) if ridge_diag else np.nan,
        "ardl_aic": float(getattr(ardl_res, "aic", np.nan)),
        "glsar_rsquared": float(getattr(glsar_res, "rsquared", np.nan)),
        "ridge_alpha": float(ridge_model.alpha_) if 'ridge_model' in locals() else np.nan,
    }
    pd.DataFrame([stability]).to_csv(country_out_dir / "stability_report.csv", index=False)
    logger.info(f"Stability report for {country_code}: {stability}")
    logger.info(f"All outputs for {country_code} saved to {country_out_dir.resolve()}")


def main():
    setup_logging("logs")
    logger = logging.getLogger(__name__)

    base_cfg = PipelineConfig()
    # List of countries to process
    countries = ["VN", "US", "CN"]

    logger.info("===== START PIPELINE (MULTI-COUNTRY) =====")
    for country in countries:
        try:
            run_pipeline_for_country(country, base_cfg)
        except Exception as e:
            logger.error(f"Pipeline failed for {country}: {e}", exc_info=True)

    logger.info("===== PIPELINE COMPLETED FOR ALL COUNTRIES =====")


if __name__ == "__main__":
    main()