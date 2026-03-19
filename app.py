# app.py — Segment Effect Explorer (v2)
# Run: streamlit run app.py

import ast
import io
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations as _combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_APP_DIR = Path(__file__).parent

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

try:
    from generate_dummy_data import SEGMENT_DESCRIPTIONS, SEGMENT_LABELS
except ImportError:
    SEGMENT_LABELS: Dict[str, str] = {}
    SEGMENT_DESCRIPTIONS: Dict[str, str] = {}

# Load real segment data from segment_descriptions.csv (overrides generate_dummy_data values)
try:
    _sdesc_df = pd.read_csv(_APP_DIR / "segment_descriptions.csv", dtype=str)
    if {"nsegment", "label", "description"}.issubset(_sdesc_df.columns):
        SEGMENT_LABELS = dict(zip(_sdesc_df["nsegment"], _sdesc_df["label"].fillna("")))
        SEGMENT_DESCRIPTIONS = dict(zip(_sdesc_df["nsegment"], _sdesc_df["description"].fillna("")))
        _SEGMENT_GROUPS: Dict[str, List[str]] = {}
        for _, _srow in _sdesc_df.iterrows():
            _SEGMENT_GROUPS.setdefault(_srow["label"], []).append(_srow["nsegment"])
    else:
        _SEGMENT_GROUPS: Dict[str, List[str]] = {}
except FileNotFoundError:
    _SEGMENT_GROUPS: Dict[str, List[str]] = {}

# ── Config ────────────────────────────────────────────────────────────────────
COMM_ORDER = ["day1", "day5", "day7", "day31", "day61DD", "day61NDD", "day90", "day120"]
REQUIRED_COLS = [
    "communication", "alpha_key", "contact_flag",
    "start_date", "end_date",
    "start_balance", "end_balance",
    "start_accounts", "end_accounts",
    "nsegments",
]

Z95 = 1.96   # z-score for 95% CI


# ── Helpers ───────────────────────────────────────────────────────────────────
def _try_parse_listlike(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None and str(v).strip() != ""]
    if isinstance(x, (tuple, set)):
        return [str(v) for v in list(x) if v is not None and str(v).strip() != ""]
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(v) for v in val if v is not None and str(v).strip() != ""]
        except Exception:
            pass
        if "," in s:
            parts = [p.strip().strip(chr(39)).strip(chr(34)) for p in s.split(",")]
            return [p for p in parts if p != ""]
        # dash-delimited format: "-000302-3290392-3203203-" → ["000302", "3290392", "3203203"]
        if s.startswith("-") and s.endswith("-") and len(s) > 1:
            parts = [p.strip() for p in s.strip("-").split("-")]
            return [p for p in parts if p != ""]
        return [s.strip(chr(39)).strip(chr(34))]
    return [str(x)]


def _df_hash(df: pd.DataFrame) -> int:
    """Fast approximate hash for large DataFrames — samples ~1 000 evenly-spaced rows."""
    step = max(1, len(df) // 1000)
    return int(pd.util.hash_pandas_object(df.iloc[::step]).sum()) ^ hash(df.shape)


@st.cache_data(show_spinner=False)
def _load_csv(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b))
    return df.rename(columns={"Communication": "communication", "Contact_flag": "contact_flag"})


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash})
def preprocess(df_raw: pd.DataFrame, date_min: Optional[str], date_max: Optional[str], recency_decay: float, bal_clip_pct: float = 0.0) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()  # normalise: Communication→communication, Contact_flag→contact_flag
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df["communication"] = df["communication"].astype(str).str.strip()
    df["contact_flag"]  = pd.to_numeric(df["contact_flag"], errors="coerce").fillna(0).astype(int)
    # Use explicit control_flag column from data if present; otherwise infer from contact_flag
    if "control_flag" in df.columns:
        df["control_flag"] = pd.to_numeric(df["control_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["control_flag"] = (df["contact_flag"] == 0).astype(int)
    for col in ["start_balance", "end_balance", "start_accounts", "end_accounts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date range filter
    if date_min:
        df = df[df["start_date"] >= pd.Timestamp(date_min)]
    if date_max:
        df = df[df["start_date"] <= pd.Timestamp(date_max)]

    # Normalise dash-delimited strings like "-000302-3290392-" into a list before further parsing
    df["nsegments"] = df["nsegments"].apply(_try_parse_listlike)
    df["nsegments"] = df["nsegments"].apply(lambda xs: xs if xs else ["__NO_SEGMENT__"])
    df = df.explode("nsegments").rename(columns={"nsegments": "nsegment"})
    df["nsegment"] = df["nsegment"].astype(str)

    eps = 1e-6
    df["balance_abs_change"] = df["end_balance"] - df["start_balance"]
    denom_b = df["start_balance"].abs().replace(0, np.nan)
    df["balance_pct_change"] = df["balance_abs_change"] / denom_b
    df.loc[df["start_balance"].abs() < eps, "balance_pct_change"] = np.nan

    df["accounts_abs_change"] = df["end_accounts"] - df["start_accounts"]
    denom_a = df["start_accounts"].replace(0, np.nan)
    df["accounts_pct_change"] = df["accounts_abs_change"] / denom_a
    df.loc[df["start_accounts"].fillna(0) == 0, "accounts_pct_change"] = np.nan

    # Outlier clipping on balance % change
    if bal_clip_pct > 0:
        lo = df["balance_pct_change"].quantile(bal_clip_pct / 100.0)
        hi = df["balance_pct_change"].quantile(1.0 - bal_clip_pct / 100.0)
        df["balance_pct_change"] = df["balance_pct_change"].clip(lo, hi)

    # Recency weights: exponential decay from newest observation
    if recency_decay > 0 and df["start_date"].notna().any():
        ref = df["start_date"].max()
        age_days = (ref - df["start_date"]).dt.days.fillna(0).clip(lower=0)
        df["_weight"] = np.exp(-recency_decay * age_days / 365.0)
    else:
        df["_weight"] = 1.0

    return df


def _rank(c: str) -> int:
    try:
        return COMM_ORDER.index(c)
    except ValueError:
        return 999


def _wmean(vals: pd.Series, weights: pd.Series) -> float:
    mask = vals.notna()
    v, w = vals[mask], weights[mask]
    if w.sum() == 0:
        return np.nan
    return float(np.average(v, weights=w))


def _ci95(vals: pd.Series, weights: pd.Series) -> float:
    """Half-width of 95% CI using weighted standard error × 1.96."""
    mask = vals.notna()
    v = vals[mask].to_numpy(dtype=float)
    w = weights[mask].to_numpy(dtype=float)
    n = len(v)
    if n < 2 or w.sum() == 0:
        return np.nan
    wm = float(np.average(v, weights=w))
    wvar = float(np.average((v - wm) ** 2, weights=w)) * n / (n - 1)
    return float(Z95 * np.sqrt(wvar / n))


def _se(vals: pd.Series, weights: pd.Series) -> float:
    """Weighted standard error of the mean (= ci95 / Z95)."""
    mask = vals.notna()
    v = vals[mask].to_numpy(dtype=float)
    w = weights[mask].to_numpy(dtype=float)
    n = len(v)
    if n < 2 or w.sum() == 0:
        return np.nan
    wm = float(np.average(v, weights=w))
    wvar = float(np.average((v - wm) ** 2, weights=w)) * n / (n - 1)
    return float(np.sqrt(wvar / n))


@st.cache_data(show_spinner=False)
def _all_segment_ids(df_raw: pd.DataFrame) -> List[str]:
    segs: set = set()
    for v in df_raw["nsegments"]:
        for s in _try_parse_listlike(v):
            if s and s != "__NO_SEGMENT__":
                segs.add(s)
    return sorted(segs)


@st.fragment
def _seg_lookup_widget(seg_ids: List[str], labels: Dict[str, str], descriptions: Dict[str, str]) -> None:
    """Searchable segment lookup — Streamlit native search handles filtering as the user types."""
    sel = st.selectbox(
        "Segment lookup",
        options=seg_ids,
        index=None,
        placeholder="Type segment ID or name…",
        key="seg_lookup_sel",
        format_func=lambda x: f"{x}  —  {labels.get(x, '')}",
    )
    if sel:
        lbl  = labels.get(sel, "")
        desc = descriptions.get(sel, "")
        if lbl:
            st.caption(f"**{lbl}** — {desc}" if desc else f"**{lbl}**")
        elif desc:
            st.caption(desc)
    st.divider()


# ── Aggregation ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash})
def agg_cols_data(
    df: pd.DataFrame,
    selected: Tuple[str, ...],
    all_mode: bool,
    bal_baseline_min: Optional[float],
) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["nsegment", "agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci", "agg_lift_bal", "agg_lift_bal_ci", "agg_lift_acct", "agg_lift_acct_ci", "agg_n"])
    if not selected:
        return EMPTY
    treated = df[(df["contact_flag"] == 1) & (df["communication"].isin(list(selected)))].copy()
    control = df[(df["control_flag"] == 1) & (df["communication"].isin(list(selected)))].copy()
    if treated.empty:
        return EMPTY
    treated["_r"] = treated["communication"].map(_rank)
    if all_mode:
        cnt = (treated.groupby(["alpha_key", "nsegment"])["communication"]
               .nunique().reset_index(name="_c"))
        ok = cnt[cnt["_c"] >= len(set(selected))][["alpha_key", "nsegment"]]
        treated = treated.merge(ok, on=["alpha_key", "nsegment"], how="inner")
        if treated.empty:
            return EMPTY
    treated = (treated.sort_values("_r", ascending=False)
                      .drop_duplicates(subset=["alpha_key", "nsegment"], keep="first"))
    # Low-balance filter: exclude segments with mean start_balance below minimum
    if bal_baseline_min is not None:
        seg_base = treated.groupby("nsegment")["start_balance"].mean()
        ok_segs = seg_base[seg_base >= bal_baseline_min].index
        treated = treated[treated["nsegment"].isin(ok_segs)]
    if treated.empty:
        return EMPTY

    # ── Vectorised per-segment stats ─────────────────────────────────────────
    g_t      = treated.groupby("nsegment", dropna=False)
    cnt_b_t  = g_t["balance_pct_change"].count()
    cnt_a_t  = g_t["accounts_pct_change"].count()
    mean_b_t = g_t["balance_pct_change"].mean()
    mean_a_t = g_t["accounts_pct_change"].mean()
    std_b_t  = g_t["balance_pct_change"].std(ddof=0)
    std_a_t  = g_t["accounts_pct_change"].std(ddof=0)
    se_b_t   = (std_b_t / np.sqrt(cnt_b_t)).where(cnt_b_t >= 2)
    se_a_t   = (std_a_t / np.sqrt(cnt_a_t)).where(cnt_a_t >= 2)
    n_u      = g_t["alpha_key"].nunique()

    if not control.empty:
        g_c      = control.groupby("nsegment", dropna=False)
        mean_b_c = g_c["balance_pct_change"].mean().reindex(mean_b_t.index)
        mean_a_c = g_c["accounts_pct_change"].mean().reindex(mean_a_t.index)
        cnt_b_c  = g_c["balance_pct_change"].count().reindex(mean_b_t.index)
        cnt_a_c  = g_c["accounts_pct_change"].count().reindex(mean_a_t.index)
        std_b_c  = g_c["balance_pct_change"].std(ddof=0).reindex(mean_b_t.index)
        std_a_c  = g_c["accounts_pct_change"].std(ddof=0).reindex(mean_a_t.index)
        se_b_c   = (std_b_c / np.sqrt(cnt_b_c)).where(cnt_b_c >= 2)
        se_a_c   = (std_a_c / np.sqrt(cnt_a_c)).where(cnt_a_c >= 2)
        lift_b    = mean_b_t - mean_b_c
        lift_a    = mean_a_t - mean_a_c
        lift_b_ci = Z95 * np.sqrt(se_b_t**2 + se_b_c**2)
        lift_a_ci = Z95 * np.sqrt(se_a_t**2 + se_a_c**2)
    else:
        lift_b = lift_a = pd.Series(np.nan, index=mean_b_t.index)
        lift_b_ci = lift_a_ci = pd.Series(np.nan, index=mean_b_t.index)

    out = pd.DataFrame({
        "nsegment":         mean_b_t.index,
        "agg_bal":          mean_b_t.values,
        "agg_bal_ci":       (Z95 * se_b_t).values,
        "agg_acct":         mean_a_t.values,
        "agg_acct_ci":      (Z95 * se_a_t).values,
        "agg_lift_bal":     lift_b.values,
        "agg_lift_bal_ci":  lift_b_ci.values,
        "agg_lift_acct":    lift_a.values,
        "agg_lift_acct_ci": lift_a_ci.values,
        "agg_n":            n_u.values,
    })
    return out if not out.empty else EMPTY


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash})
def comm_data(df: pd.DataFrame, comm: str) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["nsegment", f"{comm}_bal", f"{comm}_bal_ci", f"{comm}_acct", f"{comm}_acct_ci", f"{comm}_n", f"{comm}_lift_bal", f"{comm}_lift_bal_ci", f"{comm}_lift_acct", f"{comm}_lift_acct_ci"])
    treated = df[(df["contact_flag"] == 1) & (df["communication"] == comm)]
    control = df[(df["control_flag"] == 1) & (df["communication"] == comm)]
    if treated.empty:
        return EMPTY

    def _vstat(sub):
        """Vectorised per-segment mean, biased-SE (weights always 1.0 since recency_decay=0)."""
        g      = sub.groupby("nsegment", dropna=False)
        cnt_b  = g["balance_pct_change"].count()
        cnt_a  = g["accounts_pct_change"].count()
        mean_b = g["balance_pct_change"].mean()
        mean_a = g["accounts_pct_change"].mean()
        std_b  = g["balance_pct_change"].std(ddof=0)
        std_a  = g["accounts_pct_change"].std(ddof=0)
        # SE = biased-std / sqrt(n); NaN when fewer than 2 valid observations
        se_b   = (std_b / np.sqrt(cnt_b)).where(cnt_b >= 2)
        se_a   = (std_a / np.sqrt(cnt_a)).where(cnt_a >= 2)
        return mean_b, mean_a, se_b, se_a

    t_mb, t_ma, t_se_b, t_se_a = _vstat(treated)
    n_u = treated.groupby("nsegment", dropna=False)["alpha_key"].nunique()

    if not control.empty:
        c_mb, c_ma, c_se_b, c_se_a = _vstat(control)
        lift_b    = t_mb - c_mb.reindex(t_mb.index)
        lift_a    = t_ma - c_ma.reindex(t_ma.index)
        lift_b_ci = Z95 * np.sqrt(t_se_b**2 + c_se_b.reindex(t_mb.index)**2)
        lift_a_ci = Z95 * np.sqrt(t_se_a**2 + c_se_a.reindex(t_ma.index)**2)
    else:
        lift_b = lift_a = pd.Series(np.nan, index=t_mb.index)
        lift_b_ci = lift_a_ci = pd.Series(np.nan, index=t_mb.index)

    result = pd.DataFrame({
        "nsegment":             t_mb.index,
        f"{comm}_bal":          t_mb.values,
        f"{comm}_bal_ci":       (Z95 * t_se_b).values,
        f"{comm}_acct":         t_ma.values,
        f"{comm}_acct_ci":      (Z95 * t_se_a).values,
        f"{comm}_n":            n_u.reindex(t_mb.index).values,
        f"{comm}_lift_bal":     lift_b.values,
        f"{comm}_lift_bal_ci":  lift_b_ci.values,
        f"{comm}_lift_acct":    lift_a.values,
        f"{comm}_lift_acct_ci": lift_a_ci.values,
    })
    return result if not result.empty else EMPTY


# Minimum N for a lift to be statistically meaningful.
# Below this, the standard error is too large for the lift to be trusted.
# Rule of thumb: N=30 gives ~18% standard error on a typical proportion;
# N=50 gives ~14%. We use 30 as floor but expose it as a constant.
_MIN_N_DEFAULT = 30


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash})
def build_table(
    df: pd.DataFrame,
    ordered_comms: List[str],
    all_mode: bool,
    min_n: int,
    bal_baseline_min: Optional[float],
    _ver: int = 2,  # bump to invalidate stale cache
) -> pd.DataFrame:
    base = pd.DataFrame({"nsegment": sorted(df["nsegment"].unique())})
    a = agg_cols_data(df, tuple(ordered_comms), all_mode, bal_baseline_min)
    tbl = base.merge(a, on="nsegment", how="left")
    if "agg_n" in tbl.columns:
        mask = tbl["agg_n"].fillna(0) < min_n
        for col in ["agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci",
                    "agg_lift_bal", "agg_lift_bal_ci", "agg_lift_acct", "agg_lift_acct_ci"]:
            if col in tbl.columns:
                tbl.loc[mask, col] = np.nan
    # Fetch per-communication stats in parallel
    with ThreadPoolExecutor() as pool:
        comm_results = list(pool.map(lambda c: comm_data(df, c), ordered_comms))
    for comm, ca in zip(ordered_comms, comm_results):
        tbl = tbl.merge(ca, on="nsegment", how="left")
        nc = f"{comm}_n"
        if nc in tbl.columns:
            mask = tbl[nc].fillna(0) < min_n
            for col in [f"{comm}_bal", f"{comm}_bal_ci", f"{comm}_acct", f"{comm}_acct_ci",
                        f"{comm}_lift_bal", f"{comm}_lift_bal_ci",
                        f"{comm}_lift_acct", f"{comm}_lift_acct_ci"]:
                if col in tbl.columns:
                    tbl.loc[mask, col] = np.nan
    metric_cols = [c for c in tbl.columns if c != "nsegment" and not c.endswith("_n")]
    tbl = tbl.dropna(subset=metric_cols, how="all").set_index("nsegment")
    return tbl


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash})
def _compute_combos(
    df: pd.DataFrame,
    comm: str,
    pool_segs: tuple,   # ordered tuple of segment IDs (hashable)
    min_n: int,
    ind_lifts: tuple,   # sorted tuple of (seg, lift) pairs — hashable dict proxy
) -> list:
    """Vectorised combo-explorer computation. Cached to avoid rerunning on rerenders."""
    _ind_lifts_d = dict(ind_lifts)
    _ce_df_comm = df[df["communication"] == comm]
    _ce_treated = _ce_df_comm[_ce_df_comm["contact_flag"] == 1]
    _ce_ctrl    = _ce_df_comm[_ce_df_comm["contact_flag"] == 0]

    # Pre-build segment → set of alpha_keys (one pass per segment, not per pair)
    _ce_t_idx = {s: set(_ce_treated.loc[_ce_treated["nsegment"] == s, "alpha_key"]) for s in pool_segs}
    _ce_c_idx = {s: set(_ce_ctrl.loc[_ce_ctrl["nsegment"] == s, "alpha_key"]) for s in pool_segs}

    # Pre-build user→balance dicts — avoids O(N_users) .isin() scan for each pair
    _ce_user_bal_t = _ce_treated.groupby("alpha_key")["balance_pct_change"].first().to_dict()
    _ce_user_bal_c = _ce_ctrl.groupby("alpha_key")["balance_pct_change"].first().to_dict()

    _combo_data = []
    for _s1, _s2 in _combinations(pool_segs, 2):
        _both_t = _ce_t_idx[_s1] & _ce_t_idx[_s2]
        if len(_both_t) < min_n:
            continue
        _vals_t = np.array([_ce_user_bal_t[u] for u in _both_t if u in _ce_user_bal_t], dtype=float)
        _avg_t  = float(_vals_t.mean()) if len(_vals_t) > 0 else np.nan
        _both_c = _ce_c_idx[_s1] & _ce_c_idx[_s2]
        _vals_c = np.array([_ce_user_bal_c[u] for u in _both_c if u in _ce_user_bal_c], dtype=float)
        _avg_c  = float(_vals_c.mean()) if len(_vals_c) > 0 else np.nan
        _ce_combo_lift = (_avg_t - _avg_c) if (pd.notna(_avg_t) and pd.notna(_avg_c)) else np.nan
        _ce_best_ind   = max(float(_ind_lifts_d.get(_s1, np.nan)), float(_ind_lifts_d.get(_s2, np.nan)))
        _ce_synergy    = (_ce_combo_lift - _ce_best_ind) if (pd.notna(_ce_combo_lift) and pd.notna(_ce_best_ind)) else np.nan
        _combo_data.append({
            "Segments":       f"{_s1} + {_s2}",
            "_s1": _s1, "_s2": _s2,
            "N customers":    len(_both_t),
            "Combo Lift Bal": _ce_combo_lift,
            "Best Ind. Lift": _ce_best_ind,
            "Synergy":        _ce_synergy,
            "_lbl1":          SEGMENT_LABELS.get(_s1, ""),
            "_lbl2":          SEGMENT_LABELS.get(_s2, ""),
        })
    return _combo_data


def _rdylgn(val: float, lo: float = -0.10, hi: float = 0.10) -> str:
    if pd.isna(val):
        return ""
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    if t < 0.5:
        s = t * 2
        r, g, b = 220, int(220 * s), 50
    else:
        s = (t - 0.5) * 2
        r, g, b = int(220 * (1 - s)), 200, 50
    return f"background-color: rgba({r},{g},{b},0.55); color: #111"


def style_tbl(
    tbl: pd.DataFrame,
    ordered_comms: List[str],
    seg_labels: Optional[Dict[str, str]] = None,
    seg_desc: Optional[Dict[str, str]] = None,
    show_n_cols: bool = False,
    show_lift: bool = False,
    show_metric: str = "both",
) -> str:
    """
    Return a full HTML document with:
    - Hover tooltip on nsegment column (pure CSS :hover, not always-visible)
    - RdYlGn colour coding on % columns
    - ±CI95 column toggle
    - Lift column toggle
    """
    disp = tbl.copy()
    disp.index.name = None  # prevents pandas from rendering a second <tr> in <thead> for the index label
    orig_ids = disp.index.tolist()

    # Build column rename map
    rename = {
        "agg_bal":            "Bal%\u0394 (agg)",
        "agg_bal_ci":         "\u00b1CI Bal (agg)",
        "agg_acct":           "Acct%\u0394 (agg)",
        "agg_acct_ci":        "\u00b1CI Acct (agg)",
        "agg_lift_bal":       "Lift Bal (agg)",
        "agg_lift_bal_ci":    "\u00b1CI Lift Bal (agg)",
        "agg_lift_acct":      "Lift Acct (agg)",
        "agg_lift_acct_ci":   "\u00b1CI Lift Acct (agg)",
        "agg_n":              "N (agg)",
    }
    for c in ordered_comms:
        rename[f"{c}_bal"]          = f"{c} Bal%"
        rename[f"{c}_bal_ci"]       = f"{c} \u00b1CI"
        rename[f"{c}_acct"]         = f"{c} Acct%"
        rename[f"{c}_acct_ci"]      = f"{c} \u00b1CI Acct"
        rename[f"{c}_n"]            = f"{c} N"
        rename[f"{c}_lift_bal"]     = f"{c} Lift Bal"
        rename[f"{c}_lift_bal_ci"]  = f"{c} Lift CI±"
        rename[f"{c}_lift_acct"]    = f"{c} Lift Acct"
        rename[f"{c}_lift_acct_ci"] = f"{c} Lift Acct CI±"
    disp = disp.rename(columns=rename)

    # Drop columns based on toggles
    drop_cols = []
    # Always drop aggregate columns from table (still available in tbl for charts)
    _agg_keys = {"agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci",
                 "agg_lift_bal", "agg_lift_bal_ci", "agg_lift_acct", "agg_lift_acct_ci", "agg_n"}
    drop_cols += [v for k, v in rename.items() if k in _agg_keys and v in disp.columns]
    if not show_n_cols:
        drop_cols += [v for k, v in rename.items() if k.endswith("_n") and v in disp.columns]
    # Drop raw-% CI from table (not lift CI — that's shown when show_lift=True)
    drop_cols += [v for k, v in rename.items()
                  if ("_ci" in k) and "_lift" not in k and v in disp.columns]
    # Lift is an exclusive mode: show lift columns OR raw % columns, never both
    if show_lift:
        # drop per-comm raw % columns; keep per-comm lift + lift CI columns
        drop_cols += [v for k, v in rename.items()
                      if v in disp.columns and "_ci" not in k and "_lift" not in k
                      and (k.endswith("_bal") or k.endswith("_acct"))]
    else:
        # drop per-comm lift columns (including lift CI); show raw % columns
        drop_cols += [v for k, v in rename.items() if ("_lift" in k) and v in disp.columns]
    # Metric filter: hide balance or accounts columns based on show_metric
    if show_metric == "balance":
        drop_cols += [v for k, v in rename.items() if "_acct" in k and v in disp.columns]
    elif show_metric == "accounts":
        drop_cols += [v for k, v in rename.items() if "_bal" in k and v in disp.columns]
    disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns])

    pct_cols = [v for k, v in rename.items()
                if (k.endswith("_bal") or k.endswith("_acct") or k in ("agg_bal", "agg_acct", "agg_lift_bal", "agg_lift_acct") or "_lift" in k or "_ci" in k)
                and v in disp.columns]
    n_cols = [v for k, v in rename.items() if (k.endswith("_n") or k == "agg_n") and v in disp.columns]

    def _pct_fmt(v):
        if pd.isna(v):
            return ""
        pct = v * 100
        if abs(pct) < 0.1:
            return f"{pct:.2f}%"
        if abs(pct) < 1.0:
            s = f"{pct:.1f}%"
            # guard: rounding can push e.g. 0.96% → "1.0%"; use 2dp instead
            return f"{pct:.2f}%" if s in ("1.0%", "-1.0%") else s
        return f"{pct:.0f}%"

    fmt = {col: _pct_fmt for col in pct_cols if col in disp.columns}
    fmt.update({col: "{:,.0f}" for col in n_cols if col in disp.columns})

    def _colour_col(col_series: pd.Series) -> pd.Series:
        return col_series.map(_rdylgn)

    # Colour lift columns with significance awareness (muted when CI crosses 0)
    lift_val_cols = [v for k, v in rename.items()
                     if ("_lift" in k) and "_ci" not in k and v in disp.columns]
    lift_ci_map   = {}  # display-name lift col → display-name CI col
    for k, v in rename.items():
        if "_lift" in k and "_ci" not in k and v in disp.columns:
            ci_key = k + "_ci"
            ci_v   = rename.get(ci_key, "")
            if ci_v in disp.columns:
                lift_ci_map[v] = ci_v

    # Build per-row warning lookup: set of (row_position, lift_col_name) where CI > |lift|
    _warn_cells: set = set()
    if show_lift and lift_ci_map:
        for lc, cc in lift_ci_map.items():
            for i in range(len(disp)):
                lv = disp[lc].iat[i]
                cv = disp[cc].iat[i]
                if not pd.isna(lv) and not pd.isna(cv) and abs(cv) > abs(lv):
                    _warn_cells.add((i, lc))

    plain_colour_cols = [v for k, v in rename.items()
                         if (k.endswith("_bal") or k.endswith("_acct")
                             or k in ("agg_bal", "agg_acct"))
                         and "_lift" not in k and "_ci" not in k
                         and v in disp.columns]

    styler = disp.style.format(fmt, na_rep="")
    if plain_colour_cols:
        styler = styler.apply(_colour_col, subset=plain_colour_cols, axis=0)
    # Plain coloring for all lift value columns
    if lift_val_cols:
        styler = styler.apply(_colour_col, subset=lift_val_cols, axis=0)
    if n_cols:
        styler = styler.apply(lambda s: s.map(_n_color), subset=n_cols, axis=0)

    table_html = styler.to_html()

    # Inject ⚠️ into lift cells where CI crosses zero
    if _warn_cells:
        import re as _re_warn
        _col_names = list(disp.columns)
        _lift_col_positions = {}
        for lc in lift_ci_map:
            if lc in _col_names:
                _lift_col_positions[lc] = _col_names.index(lc)
        if _lift_col_positions:
            _tbody_parts = table_html.split("<tbody>", 1)
            if len(_tbody_parts) == 2:
                _rows = _tbody_parts[1].split("</tr>")
                _row_idx = 0
                for ri in range(len(_rows)):
                    if "<tr" not in _rows[ri]:
                        continue
                    tds = list(_re_warn.finditer(r'(<td[^>]*>)(.*?)(</td>)', _rows[ri]))
                    for lc, col_pos in _lift_col_positions.items():
                        if (_row_idx, lc) in _warn_cells and col_pos < len(tds):
                            m = tds[col_pos]
                            old = m.group(0)
                            new = f'{m.group(1)}{m.group(2)} ⚠️{m.group(3)}'
                            _rows[ri] = _rows[ri].replace(old, new, 1)
                    _row_idx += 1
                table_html = _tbody_parts[0] + "<tbody>" + "</tr>".join(_rows)

    # Inject description tooltips on row-index cells
    if seg_labels or seg_desc:
        import re as _re
        def _inject_tooltip(m):
            tag_pre = m.group(1)
            cell_val = m.group(2)
            _id  = cell_val.strip()
            lbl  = (seg_labels or {}).get(_id, "")
            desc = (seg_desc   or {}).get(_id, "")
            tip  = f"{lbl} — {desc}" if lbl and desc else (lbl or desc)
            if not tip:
                return m.group(0)
            safe = tip.replace('"', "'")
            return f'{tag_pre} data-tip="{safe}">{cell_val}</th>'
        table_html = _re.sub(
            r'(<th[^>]*class="[^"]*row_heading[^"]*"[^>]*)>([^<]*)</th>',
            _inject_tooltip, table_html,
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          font-size: 12px; background: transparent; color: #fafafa; margin: 0; padding: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  thead th {{ background: #262730; color: #fafafa; padding: 6px 10px;
              text-align: left; position: sticky; top: 0; z-index: 2;
              border-bottom: 2px solid #555; white-space: nowrap; font-size: 11px; }}
  thead th:not(.blank) {{ cursor: pointer; user-select: none; }}
  thead th:not(.blank):hover {{ background: #3a3c4a; }}
  tbody td {{ padding: 4px 10px; border-bottom: 1px solid #333; white-space: nowrap; color: #111; }}
  tbody tr:hover td {{ outline: 1px solid #666; }}
  th.row_heading {{ background: #1c1e2a !important; font-size: 11px;
                    color: #aaa !important; font-weight: normal; cursor: help;
                    position: sticky; left: 0; z-index: 1;
                    min-width: 90px; padding: 4px 16px 4px 10px; }}
  th.blank {{ background: #262730 !important;
              position: sticky; left: 0; z-index: 3; min-width: 90px; }}
</style></head>
<body><div style="overflow:auto; max-height: 615px;">
{table_html}
</div><script>
(function(){{
  var ths = document.querySelectorAll('thead tr th:not(.blank)');
  var dir = {{}};
  ths.forEach(function(th, idx) {{
    th.addEventListener('click', function() {{
      var asc = !dir[idx]; dir[idx] = asc;
      var tbody = document.querySelector('tbody');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort(function(a, b) {{
        var tds_a = a.querySelectorAll('td');
        var tds_b = b.querySelectorAll('td');
        var av = (tds_a[idx] || {{innerText:''}}).innerText.replace(/[%,\u20ac]/g,'').trim();
        var bv = (tds_b[idx] || {{innerText:''}}).innerText.replace(/[%,\u20ac]/g,'').trim();
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an-bn : bn-an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});
      rows.forEach(function(r){{ tbody.appendChild(r); }});
    }});
  }});
}})();
(function(){{
  var tt=document.createElement('div');
  tt.style.cssText='position:fixed;background:#1e2030;color:#eee;font-size:11px;padding:5px 9px;border-radius:4px;border:1px solid #555;z-index:99999;pointer-events:none;display:none;max-width:340px;word-wrap:break-word;line-height:1.5;white-space:normal;box-shadow:0 2px 8px rgba(0,0,0,.5);';
  document.body.appendChild(tt);
  document.querySelectorAll('[data-tip]').forEach(function(el){{
    el.addEventListener('mouseenter',function(e){{tt.textContent=el.getAttribute('data-tip');tt.style.display='block';}});
    el.addEventListener('mousemove', function(e){{tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY+14)+'px';}});
    el.addEventListener('mouseleave',function(){{tt.style.display='none';}});
  }});
}})();
</script></body></html>"""


def _n_color(val: object) -> str:
    """Red-yellow-green background for N (sample size) cells."""
    try:
        n = float(val)  # type: ignore[arg-type]
        if n != n or n <= 0:  # NaN or zero
            return ""
        # 0 at n=30 → 1 at n=100; clamp outside
        ratio = min(1.0, max(0.0, (n - 30) / 70.0))
        if ratio < 0.5:
            t = ratio * 2
            r = int(248 + (255 - 248) * t)
            g = int(105 + (235 - 105) * t)
            b = int(107 + (132 - 107) * t)
        else:
            t = (ratio - 0.5) * 2
            r = int(255 + (99  - 255) * t)
            g = int(235 + (190 - 235) * t)
            b = int(132 + (123 - 132) * t)
        return f"background-color: rgb({r},{g},{b}); color: #000"
    except Exception:
        return ""


def _styled_html_table(
    styler,
    seg_labels: Optional[Dict[str, str]] = None,
    seg_desc: Optional[Dict[str, str]] = None,
    height: int = 400,
) -> str:
    """Convert a Pandas Styler to a themed HTML doc with segment-ID hover tooltips."""
    import re as _re2
    html = styler.to_html()
    if seg_labels or seg_desc:
        def _tip(m):
            tag_pre  = m.group(1)
            cell_val = m.group(2)
            _id  = cell_val.strip()
            lbl  = (seg_labels or {}).get(_id, "")
            desc = (seg_desc   or {}).get(_id, "")
            tip  = f"{lbl} \u2014 {desc}" if lbl and desc else (lbl or desc)
            if not tip:
                return m.group(0)
            return f'{tag_pre} data-tip="{tip.replace(chr(34), chr(39))}">{cell_val}</th>'
        html = _re2.sub(
            r'(<th[^>]*class="[^"]*row_heading[^"]*"[^>]*)>([^<]*)</th>',
            _tip, html,
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          font-size: 12px; background: transparent; color: #fafafa; margin: 0; padding: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  thead th {{ background: #262730; color: #fafafa; padding: 6px 10px;
              text-align: left; position: sticky; top: 0; z-index: 2;
              border-bottom: 2px solid #555; white-space: nowrap; font-size: 11px; }}
  tbody td {{ padding: 4px 10px; border-bottom: 1px solid #333; white-space: nowrap; color: #111; }}
  tbody tr:hover td {{ outline: 1px solid #666; }}
  th.row_heading {{ background: #1c1e2a !important; font-size: 11px;
                    color: #ccc !important; font-weight: normal; cursor: help; }}
  th.blank {{ background: #262730 !important; }}
</style></head><body>
<div style="overflow:auto; max-height:{height - 20}px;">{html}</div>
<script>(function(){{
  var tt=document.createElement('div');
  tt.style.cssText='position:fixed;background:#1e2030;color:#eee;font-size:11px;padding:5px 9px;border-radius:4px;border:1px solid #555;z-index:99999;pointer-events:none;display:none;max-width:340px;word-wrap:break-word;line-height:1.5;white-space:normal;box-shadow:0 2px 8px rgba(0,0,0,.5);';
  document.body.appendChild(tt);
  document.querySelectorAll('[data-tip]').forEach(function(el){{
    el.addEventListener('mouseenter',function(e){{tt.textContent=el.getAttribute('data-tip');tt.style.display='block';}});
    el.addEventListener('mousemove', function(e){{tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY+14)+'px';}});
    el.addEventListener('mouseleave',function(){{tt.style.display='none';}});
  }});
}})();</script>
</body></html>"""


# ── Excel export ──────────────────────────────────────────────────────────────
def build_excel(tbl: pd.DataFrame, ordered_comms: List[str]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        _drop = [c for c in tbl.columns if c.endswith("_ci") or c.startswith("agg_")]
        out = tbl.drop(columns=_drop).reset_index()
        out.to_excel(writer, sheet_name="Segments", index=False)
        wb = writer.book
        ws = writer.sheets["Segments"]
        pct_fmt  = wb.add_format({"num_format": "0.00%"})
        green_fmt = wb.add_format({"bg_color": "#63BE7B", "num_format": "0.00%"})
        red_fmt   = wb.add_format({"bg_color": "#F8696B", "num_format": "0.00%"})
        pct_keys = ["agg_bal", "agg_acct", "agg_lift_bal", "agg_lift_acct"] + \
                   [f"{c}_bal" for c in ordered_comms] + [f"{c}_acct" for c in ordered_comms]
        col_idx = {col: i for i, col in enumerate(out.columns)}
        for col in pct_keys:
            if col not in col_idx:
                continue
            ci = col_idx[col]
            col_letter = ""
            _ci_tmp = ci
            while True:
                col_letter = chr(ord("A") + _ci_tmp % 26) + col_letter
                _ci_tmp = _ci_tmp // 26 - 1
                if _ci_tmp < 0:
                    break
            ws.conditional_format(
                1, ci, len(out), ci,
                {"type": "3_color_scale", "min_color": "#F8696B",
                 "mid_color": "#FFEB84", "max_color": "#63BE7B"},
            )
            ws.set_column(ci, ci, 14, pct_fmt)
        ws.set_column(0, 0, 12)
    return buf.getvalue()


# ── PPT export ────────────────────────────────────────────────────────────────
def build_pptx(tbl: pd.DataFrame, ordered_comms: List[str]) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Title slide
    sl = prs.slides.add_slide(blank_layout)
    txb = sl.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
    tf = txb.text_frame
    tf.text = "Affinity Explorer"
    tf.paragraphs[0].runs[0].font.size = Pt(36)
    tf.paragraphs[0].runs[0].font.bold = True
    tf.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = RGBColor(0x1C, 0x1E, 0x2A)

    # Table slide (top 20 rows, key cols)
    key_cols = ["agg_bal", "agg_lift_bal", "agg_acct", "agg_n"]
    key_cols = [c for c in key_cols if c in tbl.columns]
    top20 = tbl[key_cols].dropna(subset=["agg_bal"]).sort_values("agg_bal", ascending=False).head(20)

    sl2 = prs.slides.add_slide(blank_layout)
    sl2.background.fill.solid()
    sl2.background.fill.fore_color.rgb = RGBColor(0x1C, 0x1E, 0x2A)
    txb2 = sl2.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.6))
    txb2.text_frame.text = "Top 20 Segments — Balance % Change"
    txb2.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
    txb2.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    rows_n = len(top20) + 1
    cols_n = len(key_cols) + 1
    tbl_shape = sl2.shapes.add_table(rows_n, cols_n, Inches(0.3), Inches(0.9), Inches(12.7), Inches(6.2))
    ptbl = tbl_shape.table

    hdr_fill = RGBColor(0x26, 0x27, 0x30)
    _pptx_rename = {"agg_bal": "Balance %", "agg_lift_bal": "Lift (Balance)", "agg_acct": "Accounts %", "agg_n": "Customers"}
    headers = ["Segment"] + [_pptx_rename.get(c, c.replace("_", " ").title()) for c in key_cols]
    for ci, h in enumerate(headers):
        cell = ptbl.cell(0, ci)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(9)
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
        cell.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid()
        cell.fill.fore_color.rgb = hdr_fill

    for ri, (seg, row) in enumerate(top20.iterrows(), start=1):
        ptbl.cell(ri, 0).text = str(seg)
        ptbl.cell(ri, 0).text_frame.paragraphs[0].runs[0].font.size = Pt(8)
        for ci, col in enumerate(key_cols, start=1):
            val = row[col]
            if pd.isna(val):
                ptbl.cell(ri, ci).text = ""
            elif col.endswith("_n"):
                ptbl.cell(ri, ci).text = f"{int(val):,}"
            else:
                ptbl.cell(ri, ci).text = f"{val:.2%}"
            ptbl.cell(ri, ci).text_frame.paragraphs[0].runs[0].font.size = Pt(8)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Affinity Explorer", layout="wide")

st.markdown("""
<style>
span[data-baseweb="tag"] { background-color: #31333F !important; }
span[data-baseweb="tag"] span { color: #FAFAFA !important; }
span[data-baseweb="tag"] button { color: #FAFAFA !important; }
div.block-container { padding-top: 3rem !important; }
</style>
""", unsafe_allow_html=True)
st.title("Affinity Explorer")
st.caption(
    "Analyse how each communication touchpoint affects customer balances and accounts across segments. "
    "Use the **Segment Explorer** to browse segments, **Data** for the full data grid, "
    "**Audience Simulator** for audience recommendations, **Charts** for visual deep-dives, "
    "and **Data Quality** for health checks. Adjust filters in the sidebar."
)

# ── Load raw CSV ──────────────────────────────────────────────────────────────
try:
    with open(_APP_DIR / "dummy_segment_data.csv", "rb") as f:
        raw_bytes = f.read()
    df_raw = _load_csv(raw_bytes)
except FileNotFoundError:
    st.error("dummy_segment_data.csv not found — run `python generate_dummy_data.py` first.")
    st.stop()

missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error("Missing required columns: " + ", ".join(missing_cols))
    st.stop()

# Max unique users in the dataset — used as the upper bound for the min-N slider
_max_slider_n = max(30, int(df_raw["alpha_key"].nunique()))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Segment lookup ────────────────────────────────────────────────────
    _seg_ids = _all_segment_ids(df_raw)
    _seg_lookup_widget(_seg_ids, SEGMENT_LABELS, SEGMENT_DESCRIPTIONS)

    st.subheader("Date range")
    _dates = pd.to_datetime(df_raw["start_date"], errors="coerce").dropna()
    _d_min = _dates.min().date()
    _d_max = _dates.max().date()
    _dcol1, _dcol2 = st.columns(2)
    with _dcol1:
        date_from = st.date_input("From", value=_d_min, min_value=_d_min, max_value=_d_max)
    with _dcol2:
        date_to = st.date_input("To", value=_d_max, min_value=_d_min, max_value=_d_max)

    st.divider()
    with st.expander("Advanced settings", expanded=False):
        all_mode = False
        # min_n caption
        min_n = _MIN_N_DEFAULT
        st.caption(
            f"Segments with fewer than **{min_n}** treated customers are hidden "
            f"(minimum for statistically meaningful lift estimates)."
        )
        st.markdown("---")
        bal_clip_pct = 5  # clip below 5th and above 95th percentile (hardcoded)
        st.caption("Outlier clipping: bottom and top 5% of balance changes are removed before aggregation.")
        st.markdown("---")
        _BAL_BASELINE_MIN = 25
        bal_baseline_min = float(_BAL_BASELINE_MIN)
        st.caption(f"Low-balance filter: segments with avg starting balance < €{_BAL_BASELINE_MIN:,} are excluded from rankings.")

    # Show columns — metric toggle lives inside the Data tab (see tab_table section)
    _show_bal       = True
    _show_acct      = True
    show_n_cols     = False
    _show_metric_val = "both"  # default; overridden by widget inside tab_table

    recency_decay = 0.0  # treat all dates equally


# ── Preprocess ────────────────────────────────────────────────────────────────
df = preprocess(
    df_raw,
    date_min=str(date_from),
    date_max=str(date_to),
    recency_decay=recency_decay,
    bal_clip_pct=float(bal_clip_pct),
)

# ── ordered_comms from session state (checkboxes live inside Table tab) ─────────
_all_present_comms = [c for c in COMM_ORDER if c in df["communication"].unique()]
ordered_comms = [c for c in _all_present_comms if st.session_state.get(f"cb_{c}", True)]
if not ordered_comms:
    ordered_comms = _all_present_comms[:]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_explorer, tab_table, tab_simulator, tab_charts, tab_audit = st.tabs([
    "Segment Explorer",
    "Data",
    "Audience Simulator",
    "Charts",
    "Data Quality",
])


# ══════════════════════════════════════════════════════════
# SEGMENT EXPLORER TAB
# ══════════════════════════════════════════════════════════
with tab_explorer:
    st.subheader("Segment catalogue")
    st.caption("Browse all segments by category. Unique Customers = contacted customers in the active date range.")
    if not SEGMENT_LABELS:
        st.info("No segment descriptions found. Ensure segment_descriptions.csv is present in the app directory.")
    else:
        # ── Build base table from CSV (all segments defined) ──────────────────
        _exp_base = pd.DataFrame({
            "Segment ID": list(SEGMENT_LABELS.keys()),
            "Group":      list(SEGMENT_LABELS.values()),
            "Description": [SEGMENT_DESCRIPTIONS.get(k, "") for k in SEGMENT_LABELS],
        })
        _user_counts = (
            df[df["contact_flag"] == 1]
            .groupby("nsegment")["alpha_key"]
            .nunique()
            .rename("Unique Customers")
            .reset_index()
            .rename(columns={"nsegment": "Segment ID"})
        )
        _exp_base = _exp_base.merge(_user_counts, on="Segment ID", how="left")
        _exp_base["Unique Customers"] = _exp_base["Unique Customers"].fillna(0).astype(int)
        _exp_base = _exp_base.sort_values("Segment ID").reset_index(drop=True)

        # ── Full segment table (always visible) ────────────────────────────────
        _exp_styler = (
            _exp_base.set_index("Segment ID")
            .style
            .apply(lambda s: s.map(_n_color), subset=["Unique Customers"], axis=0)
            .format(na_rep="")
        )
        _exp_h = max(300, min(700, 60 + len(_exp_base) * 26))
        components.html(
            _styled_html_table(_exp_styler, SEGMENT_LABELS, SEGMENT_DESCRIPTIONS, height=_exp_h),
            height=_exp_h, scrolling=True,
        )

        st.divider()

        # ── Group size chart ───────────────────────────────────────────────────────
        _grp_sz = (
            _exp_base.groupby("Group")
            .agg(Segments=("Segment ID", "count"), Customers=("Unique Customers", "sum"))
            .reset_index()
            .sort_values("Customers", ascending=False)
        )
        _fig_gsz = px.bar(
            _grp_sz, x="Group", y="Segments",
            color="Segments", color_continuous_scale="Blues",
            text="Segments",
            title="Segments per group",
        )
        _fig_gsz.update_xaxes(tickangle=-30)
        _fig_gsz.update_traces(textposition="outside")
        _fig_gsz.update_layout(
            coloraxis_showscale=False, height=360,
            yaxis_range=[0, int(_grp_sz["Segments"].max()) * 1.20],
        )
        st.plotly_chart(_fig_gsz, use_container_width=True)

        _fig_gus = px.bar(
            _grp_sz, x="Group", y="Customers",
            color="Customers", color_continuous_scale="Teal",
            text=_grp_sz["Customers"].map(lambda v: f"{v:,}"),
            title="Unique customers per group (active date range)",
        )
        _fig_gus.update_xaxes(tickangle=-30)
        _fig_gus.update_traces(textposition="outside")
        _fig_gus.update_layout(
            coloraxis_showscale=False, height=360,
            yaxis_range=[0, int(_grp_sz["Customers"].max()) * 1.20],
        )
        st.plotly_chart(_fig_gus, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TABLE TAB
# ══════════════════════════════════════════════════════════
with tab_table:
    with st.expander("How to read this table", expanded=False):
        st.markdown("""
> All % changes are measured over the **7-day window** between start_date and end_date. Click any column header to sort. Hover over a segment ID for its description.

**Columns** show each communication touchpoint (day1, day5, …). Each cell is the **mean % change** in balance (or accounts) for customers in that segment who received that communication.

**Bal%** = `(end_balance − start_balance) / start_balance` measured over the 7-day window.  
**Acct%** = `(end_accounts − start_accounts) / start_accounts` over the same window.

**Lift** (enable with the toggle) = treatment group mean minus control group mean in the same segment.  
Positive lift = the communication *added* value beyond what would have happened anyway.  
Strong colour = statistically significant (95% CI does not cross zero). Muted = inconclusive.

**Colour scale**: Red = negative, Yellow = near zero, Green = positive. Scale is relative to the visible data range.

**Blank cells** = fewer customers than the minimum-N filter, or no data for that combination.  
**Click any column header** to sort the table by that metric.
        """)
    # ── Metric toggle ─────────────────────────────────────────────────────────
    _metric_opts = ["Balance & Accounts", "Balance only", "Accounts only"]
    _mc1, _mc2 = st.columns([3, 1])
    with _mc1:
        _show_metric_radio = st.radio(
            "Show columns",
            _metric_opts,
            horizontal=True,
            key="show_metric_radio",
            label_visibility="visible",
        )
    with _mc2:
        show_n_cols = st.checkbox("Show sample size (N) columns", value=False, key="show_n_cols")
    _show_metric_map = {"Balance & Accounts": "both", "Balance only": "balance", "Accounts only": "accounts"}
    _show_metric_val = _show_metric_map[_show_metric_radio]

    # ── Row 1: comm toggle strip + show lift at right ─────────────────────────
    _n_pc = len(_all_present_comms)
    _row1 = st.columns(_n_pc + 1)
    for _i, _c in enumerate(_all_present_comms):
        _row1[_i].checkbox(_c, value=True, key=f"cb_{_c}")
    show_lift = _row1[_n_pc].checkbox(
        "Show lift vs control",
        value=True,
        key="tbl_show_lift",
        help="Lift = treatment group mean − control group mean in the same segment. "
             "Positive = the communication added value beyond background trends. "
             "Enable this to see whether the % change is *caused by* the communication, not just correlated.",
    )
    ordered_comms = [c for c in _all_present_comms if st.session_state.get(f"cb_{c}", True)]
    if not ordered_comms:
        ordered_comms = _all_present_comms[:]

    # ── Row 2 placeholder (Show columns moved to sidebar Advanced filters) ────

    with st.spinner("Computing..."):
        tbl = build_table(df, ordered_comms, all_mode, min_n, bal_baseline_min, _ver=2)

    if tbl.empty:
        st.warning("No segments pass the current filters. Try adjusting filters in the sidebar.")
    else:
        html_tbl = style_tbl(
            tbl, ordered_comms,
            seg_labels=SEGMENT_LABELS,
            seg_desc=SEGMENT_DESCRIPTIONS,
            show_n_cols=show_n_cols,
            show_lift=show_lift,
            show_metric=_show_metric_val,
        )
        _tbl_h = max(300, min(680, 55 + len(tbl) * 32))
        if show_lift:
            st.caption(
                "**Blank cells** = fewer than 30 customers in that segment × communication (too small to trust)."
            )
        components.html(html_tbl, height=_tbl_h, scrolling=True)
        if show_lift:
            with st.expander("How to read Lift \u00b1 CI", expanded=False):
                st.markdown(
                    "<p style='font-size:1.05rem;line-height:1.7'>"
                    "<b>Segment A</b> — Lift 5.0% \u00b1 2.0% → range 3.0%–7.0%. "
                    "The range stays <b>above zero</b> → the lift is <b>statistically reliable</b>.<br><br>"
                    "<b>Segment B</b> — Lift 2.0% \u00b1 3.5% → range −1.5%–5.5%. "
                    "The range <b>crosses zero</b> → the lift could actually be negative; "
                    "we can't be confident it's real."
                    "</p>",
                    unsafe_allow_html=True,
                )
                import plotly.graph_objects as go
                _ex = [
                    {"label": "Segment B — unreliable", "lift": 2.0, "ci": 3.5, "colour": "#F8696B"},
                    {"label": "Segment A — reliable", "lift": 5.0, "ci": 2.0, "colour": "#63BE7B"},
                ]
                _fig_ci = go.Figure()
                for e in _ex:
                    lo, hi = e["lift"] - e["ci"], e["lift"] + e["ci"]
                    _fig_ci.add_trace(go.Scatter(
                        x=[lo, hi], y=[e["label"], e["label"]],
                        mode="lines", line=dict(width=8, color=e["colour"]),
                        showlegend=False, hoverinfo="skip",
                    ))
                    _fig_ci.add_trace(go.Scatter(
                        x=[e["lift"]], y=[e["label"]],
                        mode="markers+text", marker=dict(size=14, color="#fff", line=dict(width=2, color=e["colour"])),
                        text=[f"{e['lift']:.1f}%"], textposition="top center",
                        textfont=dict(size=13, color="#fafafa"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    _fig_ci.add_annotation(
                        x=lo, y=e["label"], text=f"{lo:.1f}%", showarrow=False,
                        yshift=-18, font=dict(size=10, color="#bbb"),
                    )
                    _fig_ci.add_annotation(
                        x=hi, y=e["label"], text=f"{hi:.1f}%", showarrow=False,
                        yshift=-18, font=dict(size=10, color="#bbb"),
                    )
                _fig_ci.add_vline(x=0, line_dash="dash", line_color="#888", line_width=1.5,
                                  annotation_text="zero", annotation_position="top",
                                  annotation_font_size=10, annotation_font_color="#999")
                _fig_ci.update_layout(
                    height=200, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Balance Lift %", zeroline=False, gridcolor="#333",
                               ticksuffix="%", color="#ccc"),
                    yaxis=dict(color="#ccc"),
                    font=dict(color="#fafafa"),
                )
                st.plotly_chart(_fig_ci, use_container_width=True)

        # ── Recommended Audiences ────────────────────────────────────────────
        st.divider()
        st.markdown("### Recommended Audiences")
        st.caption(
            "Optimal segment groupings for each communication, ranked by lift. "
            "Use these as ready-made targeting lists — no manual segment selection needed."
        )
        _tbl_ra = tbl

        _ra_c1, _ra_c2, _ra_c3, _ra_c4 = st.columns([2, 2, 1, 1])
        with _ra_c1:
            ra_metric = st.radio(
                "Optimise for", ["Balance lift", "Accounts lift"],
                horizontal=True, key="ra_metric",
            )
        with _ra_c2:
            ra_comm = st.selectbox(
                "Communication", ordered_comms, key="ra_comm",
            )
        with _ra_c3:
            ra_top_n = st.number_input(
                "Top N (best lift)", min_value=1, max_value=50, value=10, step=1, key="ra_top_n",
            )
        with _ra_c4:
            ra_bot_n = st.number_input(
                "Bottom N (worst lift)", min_value=1, max_value=50, value=5, step=1, key="ra_bot_n",
            )

        ra_suffix = "_lift_bal" if ra_metric == "Balance lift" else "_lift_acct"
        ra_label  = "Balance Lift %" if ra_metric == "Balance lift" else "Accounts Lift %"

        def _pct_fmt_ra(v):
            if pd.isna(v): return "—"
            pct = v * 100
            if abs(pct) < 0.1:
                return f"{pct:.2f}%"
            if abs(pct) < 1.0:
                s = f"{pct:.1f}%"
                return f"{pct:.2f}%" if s in ("1.0%", "-1.0%") else s
            return f"{pct:.0f}%"

        _ra_shown_segs: List[str] = []  # populated by RA branch below

        _ra_col    = f"{ra_comm}{ra_suffix}"
        _ra_ci_col = f"{ra_comm}{'_lift_bal_ci' if ra_metric == 'Balance lift' else '_lift_acct_ci'}"
        _ra_n_col  = f"{ra_comm}_n"
        if _ra_col in _tbl_ra.columns:
            _sorted_comm = _tbl_ra[_ra_col].dropna().sort_values(ascending=False)
            _ra_top    = _sorted_comm.head(int(ra_top_n))
            _ra_bot    = _tbl_ra[_ra_col].dropna().sort_values(ascending=True).head(int(ra_bot_n))
            _ra_n_vals = _tbl_ra.loc[_ra_top.index, _ra_n_col].fillna(0) if _ra_n_col in _tbl_ra.columns else pd.Series(1.0, index=_ra_top.index)
            _ra_total_n = _ra_n_vals.sum()
            _ra_w_lift  = float((_ra_top * _ra_n_vals).sum() / _ra_total_n) if _ra_total_n > 0 else np.nan
            _ra_n_dict  = _ra_n_vals.to_dict()
            _ra_users   = df[
                (df["contact_flag"] == 1)
                & (df["communication"] == ra_comm)
                & (df["nsegment"].isin(_ra_top.index))
            ]["alpha_key"].nunique()
            _rm1, _rm2 = st.columns(2)
            _rm1.metric("Recommended audience", f"{_ra_users:,} customers")
            _rm2.metric(f"Expected {ra_label}", f"{_ra_w_lift:.2%}" if pd.notna(_ra_w_lift) else "—")
            _ci_label = f"{ra_label} ±CI"

            def _make_single_comm_table(series):
                _ci_v = _tbl_ra.loc[series.index, _ra_ci_col].values if _ra_ci_col in _tbl_ra.columns else [np.nan] * len(series)
                _n_v  = [int(_tbl_ra.at[s, _ra_n_col]) if _ra_n_col in _tbl_ra.columns and s in _tbl_ra.index else 0 for s in series.index]
                _d = pd.DataFrame({
                    "Label":   [SEGMENT_LABELS.get(str(s), "") for s in series.index],
                    ra_label:  series.values,
                    _ci_label: _ci_v,
                    "N":       _n_v,
                }, index=series.index)
                _d.index.name = None
                return _d.style.format({ra_label: _pct_fmt_ra, _ci_label: _pct_fmt_ra, "N": "{:,.0f}"}, na_rep="")\
                    .apply(lambda s: s.map(_rdylgn), subset=[ra_label], axis=0)\
                    .apply(lambda s: s.map(_n_color), subset=["N"], axis=0)

            # Top N
            st.caption(f"**Top {int(ra_top_n)} segments (highest lift)**")
            _ra_styler2 = _make_single_comm_table(_ra_top)
            _ra_h2 = max(300, min(600, 60 + int(ra_top_n) * 30))
            components.html(
                _styled_html_table(_ra_styler2, SEGMENT_LABELS, SEGMENT_DESCRIPTIONS, height=_ra_h2),
                height=_ra_h2, scrolling=True,
            )
            _ra_shown_segs = [str(s) for s in _ra_top.index]

            # Bottom N
            st.caption(f"**Bottom {int(ra_bot_n)} segments (lowest lift)**")
            _ra_bot_styler = _make_single_comm_table(_ra_bot)
            _ra_bot_h = max(200, min(500, 60 + int(ra_bot_n) * 30))
            components.html(
                _styled_html_table(_ra_bot_styler, SEGMENT_LABELS, SEGMENT_DESCRIPTIONS, height=_ra_bot_h),
                height=_ra_bot_h, scrolling=True,
            )
        else:
            st.info(f"No {ra_label} data available for **{ra_comm}**.")

        # ── Send recommended segments to Simulator ───────────────────────
        if _ra_shown_segs:
            _, _ra_btn_col, _ = st.columns([1, 2, 1])
            if _ra_btn_col.button("📌 Send to Audience Simulator", key="ra_send_sim", use_container_width=True):
                # Send exactly the recommended segments as-is (already filtered by OR/AND/NOT pool)
                st.session_state["sim_segs"]          = _ra_shown_segs
                st.session_state["sim_segs_and"]      = []   # clear — pool already applied AND
                st.session_state["sim_segs_excl"]     = []   # clear — pool already applied NOT
                st.session_state["sim_run_triggered"] = True  # auto-run on arrival
                st.success(f"Sent {len(_ra_shown_segs)} segment{'s' if len(_ra_shown_segs) != 1 else ''} to the Audience Simulator tab.")

        # ── Segment Combo Explorer ────────────────────────────────────────────
        st.divider()
        st.markdown("### Segment Combo Explorer")
        st.caption(
            "Finds segment pairs that **amplify** each other — customers in both segments respond "
            "better than the best individual segment alone. "
            "**Synergy** = combo lift − best individual lift. "
            "Positive = combining these two segments unlocks extra response beyond targeting either alone."
        )
        _combo_comm = st.selectbox(
            "Communication to analyse",
            ordered_comms,
            key="combo_comm",
        )
        # Internal constants — not exposed to users
        _COMBO_POOL_N = 50   # top-N segments to scan pairs from
        _COMBO_MIN_N  = 15   # minimum shared customers to include a pair

        _lift_col_ce = f"{_combo_comm}_lift_bal"
        if _lift_col_ce not in tbl.columns:
            st.info("No lift data available for this communication.")
        else:
            _ce_pool = (
                tbl[_lift_col_ce].dropna()
                .sort_values(ascending=False)
                .head(_COMBO_POOL_N)
                .index.astype(str)
                .tolist()
            )
            _ce_ind_lifts = tbl.loc[tbl.index.isin(_ce_pool), _lift_col_ce].to_dict()
            # Pass ind_lifts as a sorted tuple of pairs so it is hashable for cache keying
            _combo_data = _compute_combos(
                df,
                _combo_comm,
                tuple(_ce_pool),
                _COMBO_MIN_N,
                tuple(sorted(_ce_ind_lifts.items())),
            )

            if _combo_data:
                _combo_df = pd.DataFrame(_combo_data).sort_values("Synergy", ascending=False)
                # Scatter: X = combo lift, Y = synergy, size = N customers
                _ce_fig = px.scatter(
                    _combo_df,
                    x="Combo Lift Bal",
                    y="Synergy",
                    size="N customers",
                    color="Synergy",
                    color_continuous_scale="RdYlGn",
                    hover_name="Segments",
                    custom_data=["_lbl1", "_lbl2", "N customers", "Best Ind. Lift"],
                    title=f"Segment pair synergy — {_combo_comm}",
                    labels={"Combo Lift Bal": "Combo Balance Lift", "Synergy": "Synergy (above best alone)"},
                )
                _ce_fig.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>"
                        "%{customdata[0]}<br>%{customdata[1]}<br>"
                        "N: %{customdata[2]:,}<br>"
                        "Combo lift: %{x:.1%}<br>"
                        "Best alone: %{customdata[3]:.1%}<br>"
                        "Synergy: %{y:.1%}<extra></extra>"
                    )
                )
                _ce_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5,
                                  annotation_text="Synergy = 0", annotation_position="bottom right")
                _ce_fig.update_xaxes(tickformat=".0%")
                _ce_fig.update_yaxes(tickformat=".0%")
                _ce_fig.update_layout(coloraxis_showscale=False, height=420)
                st.plotly_chart(_ce_fig, use_container_width=True)

                _ce_show = _combo_df[["Segments", "N customers", "Combo Lift Bal", "Best Ind. Lift", "Synergy"]].head(20).copy()
                def _pct_ce(v): return f"{v*100:.1f}%" if pd.notna(v) else "—"
                st.dataframe(
                    _ce_show.style
                    .format({"Combo Lift Bal": _pct_ce, "Best Ind. Lift": _pct_ce, "Synergy": _pct_ce, "N customers": "{:,.0f}"})
                    .background_gradient(subset=["Combo Lift Bal", "Synergy"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
                    use_container_width=True,
                )
                st.caption(f"Scanning top {_COMBO_POOL_N} segments by lift. Minimum {_COMBO_MIN_N} shared customers per pair. Top 20 shown.")
            else:
                st.info(f"No segment pairs with ≥{_COMBO_MIN_N} shared customers found in the current top {_COMBO_POOL_N} pool.")


# ══════════════════════════════════════════════════════════
# CHARTS TAB
# ══════════════════════════════════════════════════════════
with tab_charts:
    with st.expander("How to read the charts", expanded=False):
        st.markdown("""
**Bar chart** — ranks segments by the selected metric. Taller = stronger average effect. Error bars are 95% confidence intervals (wider = less certain). Use this to decide which segments to prioritise in the next campaign.

**Heatmap** — segment × communication grid. Each cell = mean % change. Dark green = strong positive response. Use this to spot *which communications* work for *which segments*, and identify segments that respond early vs late in the journey.

**Journey timeline** — tracks how selected segments perform at each touchpoint over time. Rising lines = improving engagement. Flat or falling lines = fatigue or declining relevance at that stage.

**Violin chart** — shows the *full distribution* of individual % changes, not just the average. Wide violin = high variability (average driven by a few extreme responders). Narrow = consistent. Use this to spot segments where the mean is misleading.

**KDE density explorer** — smoothed density curve, useful for comparing the *shape* of responses across segments. Separated peaks = meaningfully different groups. Overlapping peaks = similar behaviour.

**Co-occurrence heatmap** — cell [A, B] = fraction of segment A customers who are *also* in segment B. Dark blue = heavy overlap. If two high-performing segments overlap heavily, targeting both wastes budget.
        """)

    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Data tab.")
    else:
        # ── Heatmap ─────────────────────────────────────────────────────────
        _hm_metric = st.radio(
            "Heatmap metric",
            ["Balance %", "Accounts %"],
            horizontal=True,
            key="hm_metric",
            help="Balance % = average % change in account balance over the 7-day window. "
                 "Accounts % = average % change in number of open accounts. "
                 "Green cells = positive response to that communication. Red = negative.",
        )
        _hm_suffix = "_bal" if _hm_metric == "Balance %" else "_acct"
        _hm_label  = "Bal% Δ" if _hm_metric == "Balance %" else "Acct% Δ"
        st.subheader(f"{_hm_label} — segment × communication")
        st.caption(
            "**Heatmap** — each cell is the avg % change for a segment (row) "
            "at a single communication touchpoint (column). Darker green = stronger positive response. "
            "Use this to spot which communications resonate with which segments, and identify "
            "segments that respond early vs late in the journey."
        )
        heat_cols = [f"{c}{_hm_suffix}" for c in ordered_comms if f"{c}{_hm_suffix}" in tbl.columns]
        if heat_cols:
            hm = tbl[heat_cols].copy()
            hm.columns = [c.replace(_hm_suffix, "") for c in heat_cols]
            hm["_mean"] = hm.mean(axis=1)
            hm = hm.sort_values("_mean", ascending=False).head(50).drop(columns="_mean")
            hm.index = hm.index.astype(str)
            fig_hm = px.imshow(
                hm * 100,
                labels=dict(x="Communication", y="Segment", color=_hm_label),
                color_continuous_scale="RdYlGn", aspect="auto",
                title=f"{_hm_label} — Top 50 segments",
                text_auto=".1f",
            )
            _hm_custom = np.empty(hm.shape, dtype=object)
            for _ri, _sid in enumerate(hm.index):
                _lbl  = SEGMENT_LABELS.get(str(_sid), "")
                _desc = SEGMENT_DESCRIPTIONS.get(str(_sid), "")
                _row_tip = (f"<b>{_sid}</b>" + (f"<br>{_lbl}" if _lbl else "") + (f"<br><i>{_desc}</i>" if _desc else ""))
                for _ci in range(len(hm.columns)):
                    _hm_custom[_ri, _ci] = _row_tip
            fig_hm.update_traces(
                customdata=_hm_custom,
                hovertemplate="%{customdata}<br>Comm: %{x} → %{z:.1f}%<extra></extra>",
            )
            fig_hm.update_yaxes(
                type="category", autorange="reversed",
                tickvals=list(hm.index),
                ticktext=[str(s) for s in hm.index],
            )
            fig_hm.update_xaxes(type="category")
            fig_hm.update_layout(height=max(420, len(hm) * 14 + 100))
            st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ── Journey timeline ─────────────────────────────────────────────────
        st.subheader("Journey Timeline — avg % change at each touchpoint")
        st.caption(
            "**Journey timeline** — tracks how selected segments perform at each "
            "communication touchpoint. Rising lines = improving engagement over the journey. "
            "Flat or falling lines suggest fatigue or declining relevance at that stage. "
            "Useful for optimising send timing and dropping ineffective touchpoints."
        )
        _jt_metric = st.radio("Journey metric", ["Balance %", "Accounts %"], horizontal=True, key="jt_metric")
        _jt_col    = "balance_pct_change" if _jt_metric == "Balance %" else "accounts_pct_change"
        _jt_ylabel = "Avg Balance % Change" if _jt_metric == "Balance %" else "Avg Accounts % Change"
        _jt_sfx      = "_bal" if _jt_metric == "Balance %" else "_acct"
        _jt_sort_col = next((f"{c}{_jt_sfx}" for c in ordered_comms if f"{c}{_jt_sfx}" in tbl.columns), None)
        jt_pool = (tbl.sort_values(_jt_sort_col, ascending=False).head(30).index.astype(str).tolist()
                   if _jt_sort_col else tbl.index.astype(str).tolist()[:30])
        jt_segs = st.multiselect(
            "Segments for timeline (top 30)",
            options=jt_pool,
            default=jt_pool[:5],
            format_func=lambda x: x,
            key="jt_segs",
        )
        if jt_segs:
            jt_rows = []
            for comm in ordered_comms:
                sub = df[(df["contact_flag"] == 1) & (df["communication"] == comm) & (df["nsegment"].isin(jt_segs))]
                for seg in jt_segs:
                    grp = sub[sub["nsegment"] == seg]
                    if not grp.empty:
                        jt_rows.append({
                            "Communication": comm,
                            "Segment": seg,
                            _jt_ylabel: grp[_jt_col].mean(),
                            "_rank": _rank(comm),
                        })
            if jt_rows:
                jt_df = pd.DataFrame(jt_rows).sort_values("_rank")
                jt_df["_seg_tip"] = jt_df["Segment"].map(
                    lambda s: (SEGMENT_LABELS.get(s, s) + (" \u2014 " + SEGMENT_DESCRIPTIONS.get(s, "") if SEGMENT_DESCRIPTIONS.get(s) else ""))
                )
                fig_jt = px.line(
                    jt_df, x="Communication", y=_jt_ylabel, color="Segment",
                    markers=True,
                    custom_data=["Segment", "_seg_tip"],
                    category_orders={"Communication": ordered_comms},
                    title=f"{_jt_ylabel} across journey touchpoints",
                )
                fig_jt.update_traces(
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x}: %{y:.1%}<extra></extra>"
                )
                fig_jt.update_yaxes(tickformat=".1%")
                fig_jt.update_layout(height=450)
                st.plotly_chart(fig_jt, use_container_width=True)
            else:
                st.info("No journey data for selected segments.")

        st.divider()

        # ── Segment co-occurrence heatmap ────────────────────────────────────
        st.subheader("Segment Co-occurrence — how often segments share the same user")
        st.caption(
            "**Co-occurrence heatmap** — cell [A, B] = fraction of segment A customers "
            "who are also in segment B. Dark blue = heavy overlap. "
            "If two high-performing segments overlap heavily, targeting both wastes budget — "
            "pick the one with stronger lift. Also useful for building exclusion lists."
        )
        _cooc_sort = next((f"{c}_n" for c in ordered_comms if f"{c}_n" in tbl.columns), None)
        _cooc_pool_size = 100
        top_segs_cooc = (tbl.sort_values(_cooc_sort, ascending=False).head(_cooc_pool_size).index.astype(str).tolist()
                         if _cooc_sort else tbl.index.astype(str).tolist()[:_cooc_pool_size])
        if len(top_segs_cooc) < 2:
            st.info("Not enough segments with data to compute co-occurrence. Adjust filters.")
        else:
            _cooc_max = len(top_segs_cooc)
            if _cooc_max <= 2:
                cooc_n = _cooc_max
            else:
                _cooc_def = min(40, _cooc_max)
                if "cooc_n" in st.session_state:
                    st.session_state["cooc_n"] = max(2, min(int(st.session_state["cooc_n"]), _cooc_max))
                cooc_n = st.slider("Top N segments to include", 2, _cooc_max, _cooc_def, key="cooc_n")
            cooc_segs = top_segs_cooc[:cooc_n]

            user_seg_df = (
                df[df["nsegment"].isin(cooc_segs)]
                .groupby("alpha_key")["nsegment"]
                .apply(set)
                .reset_index()
            )
            n_users_total = user_seg_df["alpha_key"].nunique()
            # Vectorised co-occurrence via indicator matrix multiplication
            _indicator = (
                df[df["nsegment"].isin(cooc_segs)]
                .drop_duplicates(subset=["alpha_key", "nsegment"])
                .assign(_v=1)
                .pivot_table(index="alpha_key", columns="nsegment", values="_v", fill_value=0)
            )
            _indicator = _indicator.reindex(columns=cooc_segs, fill_value=0)
            _vals = _indicator.values  # shape (n_users, n_segs)
            cooc_matrix = pd.DataFrame(
                _vals.T @ _vals,  # (n_segs, n_segs) — counts of users sharing both segments
                index=cooc_segs, columns=cooc_segs, dtype=float,
            )
            # Normalize by diagonal (self-count = total users in segment)
            with np.errstate(divide="ignore", invalid="ignore"):
                diag = np.diag(cooc_matrix.values).copy()
                diag[diag == 0] = 1
                cooc_norm = cooc_matrix.values / diag[:, None]
            cooc_norm_df = pd.DataFrame(cooc_norm, index=cooc_segs, columns=cooc_segs)
            cooc_norm_df.index   = [str(s) for s in cooc_segs]
            cooc_norm_df.columns = [str(s) for s in cooc_segs]
            # Sort rows by max off-diagonal co-occurrence (highest overlap at top)
            _off_diag = cooc_norm_df.copy()
            np.fill_diagonal(_off_diag.values, 0)
            _row_order = _off_diag.max(axis=1).sort_values(ascending=False).index.tolist()
            cooc_norm_df = cooc_norm_df.loc[_row_order, _row_order]
            fig_cooc = px.imshow(
                cooc_norm_df,
                color_continuous_scale="Blues",
                zmin=0, zmax=1,
                labels=dict(color="Overlap %"),
                title="Segment co-occurrence (row = % of segment A customers who also belong to segment B)",
                aspect="auto",
            )
            fig_cooc.update_xaxes(type="category")
            fig_cooc.update_yaxes(type="category", autorange="reversed")
            fig_cooc.update_layout(height=600)
            st.plotly_chart(fig_cooc, use_container_width=True)
            st.caption(
                "**How to read**: cell [A, B] = fraction of segment A users who are also in segment B. "
                "High values (dark blue) mean heavy overlap — avoid targeting both segments simultaneously."
            )

        st.divider()

        # ── Distribution explorer ────────────────────────────────────────────
        st.subheader("Distribution explorer — KDE density")
        st.caption(
            "**Distribution explorer** — smoothed KDE density curve for any "
            "combination of segments and communication. Use this to compare the shape of responses "
            "side by side: overlapping peaks = similar behaviour; separated peaks = meaningfully "
            "different customer groups. Segmentation is only useful if groups behave differently."
        )
        dc1, dc2, dc3 = st.columns([3, 1, 1])
        all_segs = sorted(df["nsegment"].unique().tolist())
        with dc1:
            dist_segs = st.multiselect("Select segments", options=all_segs, default=all_segs[:5],
                                       format_func=lambda x: x, key="dist_segs")
        with dc2:
            dist_comm = st.selectbox("Communication", options=["All selected"] + ordered_comms, key="dist_comm")
        with dc3:
            dist_metric = st.radio("Metric", ["Balance %", "Accounts %"], horizontal=True, key="dist_metric")

        if dist_segs:
            dm = df[(df["contact_flag"] == 1) & (df["nsegment"].isin(dist_segs))]
            if dist_comm != "All selected":
                dm = dm[dm["communication"] == dist_comm]
            else:
                dm = dm[dm["communication"].isin(ordered_comms)]
            metric_col = "balance_pct_change" if dist_metric == "Balance %" else "accounts_pct_change"
            dm = dm.dropna(subset=[metric_col]).copy()
            if not dm.empty:
                drawn = False
                try:
                    import plotly.figure_factory as ff
                    groups  = [dm[dm["nsegment"] == s][metric_col].values.tolist() for s in dist_segs]
                    labels_ = [f"{s} — {SEGMENT_LABELS.get(s, '')}" for s in dist_segs]
                    valid   = [(g, l) for g, l in zip(groups, labels_) if len(g) >= 2]
                    if valid:
                        gv, lv = zip(*valid)
                        _colors = list(px.colors.qualitative.Bold[:len(gv)])
                        fig_dist = ff.create_distplot(list(gv), list(lv), show_hist=False, show_rug=False, colors=_colors)
                        fig_dist.update_xaxes(tickformat=".0%" if metric_col == "balance_pct_change" else ".2f", title=dist_metric)
                        fig_dist.update_layout(height=480, title=f"{dist_metric} — KDE density by segment")
                        st.plotly_chart(fig_dist, use_container_width=True)
                        drawn = True
                except Exception:
                    pass
                if not drawn:
                    dm["_label"] = dm["nsegment"].map(lambda x: f"{x} — {SEGMENT_LABELS.get(x, '')}")
                    fig_dist = px.histogram(dm, x=metric_col, color="_label", barmode="overlay", opacity=0.65,
                                            histnorm="probability density",
                                            color_discrete_sequence=px.colors.qualitative.Bold,
                                            labels={metric_col: dist_metric, "_label": "Segment"},
                                            title=f"{dist_metric} — KDE density by segment")
                    if metric_col == "balance_pct_change":
                        fig_dist.update_xaxes(tickformat=".0%")
                    fig_dist.update_layout(height=480)
                    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════
# AUDIENCE SIMULATOR TAB
# ══════════════════════════════════════════════════════════
with tab_simulator:
    st.subheader("Audience Performance Simulator")
    st.caption(
        "Select any set of segments below and we'll estimate the **expected balance lift per "
        "communication** if you sent to only those customers. "
        "Lift is N-weighted across your chosen segments: segments with more customers carry more weight."
    )

    with st.expander("How the numbers are calculated", expanded=False):
        st.markdown("""
**Lift (treatment − control)**  
For each segment × communication cell, *lift* = mean outcome for treated customers minus mean outcome for control customers in the *same segment and communication*.  
This isolates the causal effect of the communication, removing background trends that would have happened anyway.

**N-weighted expected lift**  
When you select multiple segments, the simulator computes a single headline lift by taking a weighted average across segments — where each segment's weight is its sample size (N).  
Larger segments drive the headline more. Segments with no control data are excluded from the weighted average.

**Projected absolute € increase**  
Formula: `customers × avg_start_balance × lift%`  
Example: 500 customers × €2,000 avg balance × 5% lift = €50,000 incremental increase.  
Note: this is an *expected* estimate based on historical lift — actual results will vary.

**Projected account openings**  
Same logic: `customers × avg_start_accounts × lift%`.  
A lift of 10% on an average of 1.2 accounts per user ≈ 0.12 new accounts per user targeted.
        """)

    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Data tab first.")
    else:
        all_segs_sim = sorted(tbl.index.astype(str).tolist())
        # Default to top 10 by first comm's balance lift (most actionable)
        _first_lift = f"{ordered_comms[0]}_lift_bal" if ordered_comms else None
        if _first_lift and _first_lift in tbl.columns:
            _default_sim = tbl[_first_lift].dropna().sort_values(ascending=False).head(10).index.astype(str).tolist()
        else:
            _default_sim = all_segs_sim[:min(10, len(all_segs_sim))]

        sim_segs = st.multiselect(
            "OR — segments to include (union)",
            options=all_segs_sim,
            default=st.session_state.get("sim_segs", []),
            format_func=lambda x: x,
            key="sim_segs",
            help="A customer qualifies if they belong to ANY of these segments.",
        )

        # AND multiselect — intersection filter
        sim_segs_and = st.multiselect(
            "AND — must also be in (intersection)",
            options=all_segs_sim,
            default=[s for s in st.session_state.get("sim_segs_and", []) if s in all_segs_sim],
            format_func=lambda x: x,
            key="sim_segs_and",
            help="When non-empty, only segments that appear in BOTH the OR list and this list are kept. "
                 "Use this to narrow: e.g. OR=[A,B,C] AND=[B,C,D] → effective segments are {B,C}.",
        )

        # NOT multiselect — exclusion
        sim_segs_excl = st.multiselect(
            "NOT — segments to exclude",
            options=all_segs_sim,
            default=[s for s in st.session_state.get("sim_segs_excl", []) if s in all_segs_sim],
            format_func=lambda x: x,
            key="sim_segs_excl",
            help="Any segment listed here is removed from the final set, even if it appears above.",
        )
        # Effective segments: OR ∩ AND (if set) − NOT
        _sim_segs_eff = [
            s for s in (sim_segs or [])
            if (not sim_segs_and or s in sim_segs_and)
            and s not in sim_segs_excl
        ]

        # Reset trigger if the segment selection has changed since the last run
        if _sim_segs_eff != st.session_state.get("_sim_segs_snapshot"):
            st.session_state["sim_run_triggered"] = False

        if st.button("▶ Run simulation", key="sim_run_btn", type="primary"):
            st.session_state["sim_run_triggered"] = True
            st.session_state["_sim_segs_snapshot"] = list(_sim_segs_eff)
        sim_metric = st.session_state.get("sim_metric", "Balance % lift")

        if st.session_state.get("sim_run_triggered") and _sim_segs_eff:
            # ── Pre-filter table rows and compute projected metrics ───────────
            sub_all = tbl.loc[tbl.index.isin(_sim_segs_eff)].copy()
            _sim_total_users = df[
                (df["contact_flag"] == 1) & (df["nsegment"].isin(_sim_segs_eff))
            ]["alpha_key"].nunique()

            def _w_avg(col, n_vals):
                if col not in sub_all.columns:
                    return np.nan
                valid = sub_all[col].notna()
                tot = n_vals[valid].sum()
                return float((sub_all.loc[valid, col] * n_vals[valid]).sum() / tot) if tot > 0 else np.nan

            def _safe_proj(n, avg, lift):
                if pd.isna(avg) or pd.isna(lift) or n == 0:
                    return np.nan
                return n * float(avg) * float(lift)

            sim_summary_rows = []
            for comm in ordered_comms:
                n_col = f"{comm}_n"
                if n_col not in tbl.columns:
                    continue
                n_vals  = sub_all[n_col].fillna(0) if n_col in sub_all.columns else pd.Series(dtype=float)
                total_n = int(n_vals.sum())

                w_lift_bal  = _w_avg(f"{comm}_lift_bal",  n_vals)
                w_lift_acct = _w_avg(f"{comm}_lift_acct", n_vals)
                w_raw_bal   = _w_avg(f"{comm}_bal",       n_vals)
                w_raw_acct  = _w_avg(f"{comm}_acct",      n_vals)
                w_lift = w_lift_bal  if sim_metric == "Balance % lift" else w_lift_acct
                w_raw  = w_raw_bal   if sim_metric == "Balance % lift" else w_raw_acct

                _comm_df  = df[
                    (df["contact_flag"] == 1)
                    & (df["communication"] == comm)
                    & (df["nsegment"].isin(_sim_segs_eff))
                ]
                _per_user = _comm_df.groupby("alpha_key")[["start_balance", "start_accounts"]].first()
                _comm_n_u = len(_per_user)
                _avg_sb   = _per_user["start_balance"].mean()  if not _per_user.empty else np.nan
                _avg_sa   = _per_user["start_accounts"].mean() if not _per_user.empty else np.nan
                proj_eur   = _safe_proj(_comm_n_u, _avg_sb,  w_lift_bal)
                proj_accts = _safe_proj(_comm_n_u, _avg_sa, w_lift_acct)

                sim_summary_rows.append({
                    "Communication":   comm,
                    "Expected Lift":   w_lift,
                    "Treatment Avg":   w_raw,
                    "Total N":         total_n,
                    "Comm Users":      _comm_n_u,
                    "Projected Bal \u20ac": proj_eur,
                    "Proj. Accounts":  proj_accts,
                    "_rank":           _rank(comm),
                })

            sim_summary = (pd.DataFrame(sim_summary_rows)
                           .sort_values("_rank")
                           .drop(columns="_rank")
                           .reset_index(drop=True))

            _excl_note = f" ({len(sim_segs_excl)} excluded)" if sim_segs_excl else ""
            _tot_c1, _tot_c2 = st.columns([2, 1])
            _tot_c1.metric(
                "Total unique customers (all selected segments)" + _excl_note,
                f"{_sim_total_users:,}",
                help="Unique people (alpha_key) who were contacted and appear in at least one of the selected segments.",
            )
            with _tot_c2:
                st.radio("Metric", ["Balance % lift", "Accounts % lift"], key="sim_metric", horizontal=True)
            _lift_suffix = "_lift_bal" if sim_metric == "Balance % lift" else "_lift_acct"
            _raw_suffix  = "_bal"      if sim_metric == "Balance % lift" else "_acct"
            _y_label     = "Expected Balance Lift %" if sim_metric == "Balance % lift" else "Expected Accounts Lift %"

            # Show kpi metrics row
            st.markdown("#### Expected lift across selected segments")
            kpi_cols = st.columns(len(sim_summary))
            for i, row in sim_summary.iterrows():
                lift_val = row["Expected Lift"]
                label    = row["Communication"]
                n_val    = int(row["Total N"]) if pd.notna(row["Total N"]) else 0
                u_val    = int(row["Comm Users"]) if pd.notna(row.get("Comm Users")) else 0
                delta_str = f"{n_val:,} segment-rows"
                _kpi_help = (
                    f"**Total N = {n_val:,}** — sum of per-segment sample sizes. "
                    f"A customer in {len(_sim_segs_eff)} segments counts {len(_sim_segs_eff)}×. "
                    f"**Comm Users = {u_val:,}** — unique persons who received this communication "
                    f"(used in the projection below). Both figures are correct; they measure different things."
                )
                if pd.notna(lift_val):
                    kpi_cols[i].metric(label, f"{lift_val:.2%}", delta_str, help=_kpi_help)
                else:
                    kpi_cols[i].metric(label, "—", delta_str, help=_kpi_help)

            # Projected row — only show the metric matching the selected metric
            if sim_metric == "Balance % lift":
                st.markdown("#### Projected absolute balance increase")
                proj_b_cols = st.columns(len(sim_summary))
                for i, row in sim_summary.iterrows():
                    pv = row["Projected Bal \u20ac"]
                    nu = int(row["Comm Users"]) if pd.notna(row.get("Comm Users")) else 0
                    proj_b_cols[i].metric(
                        row["Communication"],
                        f"\u20ac{pv:,.0f}" if pd.notna(pv) else "—",
                        f"{nu:,} unique customers",
                        help=(
                            f"Formula: unique customers × avg start balance × lift%. "
                            f"Uses {nu:,} *unique* persons (not segment-rows) as the headcount."
                        ),
                    )
            else:
                st.markdown("#### Projected account openings")
                proj_a_cols = st.columns(len(sim_summary))
                for i, row in sim_summary.iterrows():
                    av = row["Proj. Accounts"]
                    nu = int(row["Comm Users"]) if pd.notna(row.get("Comm Users")) else 0
                    proj_a_cols[i].metric(
                        row["Communication"],
                        f"{av:,.0f}" if pd.notna(av) else "—",
                        f"{nu:,} unique customers",
                        help=(
                            f"Formula: unique customers × avg start accounts × lift%. "
                            f"Uses {nu:,} *unique* persons as the headcount."
                        ),
                    )

            st.divider()

            # ── Per-segment detail table ──────────────────────────────────────
            st.markdown("#### Per-segment breakdown")

            # Build columns: lift, CI, and N — interleaved per comm
            _ci_suffix = "_lift_bal_ci" if sim_metric == "Balance % lift" else "_lift_acct_ci"
            _intl_cols = []
            for c in ordered_comms:
                if f"{c}{_lift_suffix}" in tbl.columns:
                    _intl_cols.append(f"{c}{_lift_suffix}")
                if f"{c}{_ci_suffix}" in tbl.columns:
                    _intl_cols.append(f"{c}{_ci_suffix}")
                if f"{c}_n" in tbl.columns:
                    _intl_cols.append(f"{c}_n")
            detail = tbl.loc[tbl.index.isin(_sim_segs_eff), [c for c in _intl_cols if c in tbl.columns]].copy()
            detail.index.name = None  # avoids blank header row in _styled_html_table

            col_rename_sim = {f"{c}{_lift_suffix}": c for c in ordered_comms}
            col_rename_sim.update({f"{c}{_ci_suffix}": f"{c} CI±" for c in ordered_comms})
            col_rename_sim.update({f"{c}_n": f"{c} N" for c in ordered_comms})
            detail = detail.rename(columns=col_rename_sim)

            def _pct_fmt_sim(v):
                if pd.isna(v): return ""
                pct = v * 100
                if abs(pct) < 0.1:
                    return f"{pct:.2f}%"
                if abs(pct) < 1.0:
                    s = f"{pct:.1f}%"
                    return f"{pct:.2f}%" if s in ("1.0%", "-1.0%") else s
                return f"{pct:.0f}%"

            pct_fmt = {c: _pct_fmt_sim for c in ordered_comms if c in detail.columns}
            pct_fmt.update({f"{c} CI±": _pct_fmt_sim for c in ordered_comms if f"{c} CI±" in detail.columns})
            n_fmt   = {f"{c} N": "{:,.0f}" for c in ordered_comms if f"{c} N" in detail.columns}
            fmt_all = {**pct_fmt, **n_fmt}

            def _lift_color(col_series):
                return col_series.map(_rdylgn)

            def _n_color_sim(col_series):
                return col_series.map(_n_color)

            lift_disp_cols = [c for c in ordered_comms if c in detail.columns]
            ci_disp_cols   = [f"{c} CI±" for c in ordered_comms if f"{c} CI±" in detail.columns]
            n_disp_cols    = [f"{c} N" for c in ordered_comms if f"{c} N" in detail.columns]
            styler_sim = detail.style.format(fmt_all, na_rep="")
            if lift_disp_cols:
                styler_sim = styler_sim.apply(_lift_color, subset=lift_disp_cols, axis=0)
            if n_disp_cols:
                styler_sim = styler_sim.apply(_n_color_sim, subset=n_disp_cols, axis=0)
            _sim_h = max(300, min(600, 60 + len(detail) * 30))
            components.html(
                _styled_html_table(styler_sim, SEGMENT_LABELS, SEGMENT_DESCRIPTIONS, height=_sim_h),
                height=_sim_h, scrolling=True,
            )

            # ── Excel export of selection ─────────────────────────────────────
            _exp_rows = [
                {
                    "nsegment":    s,
                    "label":       SEGMENT_LABELS.get(s, ""),
                    "description": SEGMENT_DESCRIPTIONS.get(s, ""),
                    "status":      "INCLUDED",
                }
                for s in _sim_segs_eff
            ] + [
                {
                    "nsegment":    s,
                    "label":       SEGMENT_LABELS.get(s, ""),
                    "description": SEGMENT_DESCRIPTIONS.get(s, ""),
                    "status":      "EXCLUDED",
                }
                for s in sim_segs_excl
            ]
            _exp_df = pd.DataFrame(_exp_rows, columns=["nsegment", "label", "description", "status"])
            _exp_buf = io.BytesIO()
            with pd.ExcelWriter(_exp_buf, engine="xlsxwriter") as _ew:
                _exp_df.to_excel(_ew, sheet_name="Selection", index=False)
                # Add summary sheet
                sim_summary.to_excel(_ew, sheet_name="Summary", index=False)
            _exp_buf.seek(0)
            st.download_button(
                "Download Excel (.xlsx)",
                _exp_buf.read(),
                file_name="targeting_selection.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Exports two sheets: 'Selection' (one row per segment with INCLUDED/EXCLUDED status) and 'Summary' (lift + projection per communication).",
            )
        elif st.session_state.get("sim_run_triggered") and not _sim_segs_eff:
            st.warning("No segments to analyse — all included segments are also in the exclude list. Remove some exclusions.")


# ══════════════════════════════════════════════════════════
# DATA QUALITY TAB
# ══════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("Data Quality & Recommendations")
    st.caption("Live health check of your dataset and current filter settings.")

    k1, k2, k3, k4 = st.columns(4)
    n_rows_raw   = len(df_raw)
    n_users_raw  = df_raw["alpha_key"].nunique() if "alpha_key" in df_raw.columns else 0
    n_segs       = df["nsegment"].nunique()
    contact_rate = df_raw["contact_flag"].mean() if "contact_flag" in df_raw.columns else 0
    k1.metric("Total records",       f"{n_rows_raw:,}")
    k2.metric("Unique customers",    f"{n_users_raw:,}")
    k3.metric("Unique segments",     f"{n_segs:,}")
    k4.metric("Contacted %",         f"{contact_rate:.1%}")

    st.divider()

    with st.expander("Data Quality", expanded=True):
        nan_bal  = df["balance_pct_change"].isna().mean()
        nan_acct = df["accounts_pct_change"].isna().mean()
        zero_bal = (df["start_balance"].fillna(0) < 1).mean()
        dup_pct  = df.duplicated(subset=["alpha_key", "communication", "nsegment"]).mean()
        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Balance data missing",    f"{nan_bal:.1%}")
        dq2.metric("Accounts data missing",   f"{nan_acct:.1%}")
        dq3.metric("Near-zero balances",       f"{zero_bal:.1%}")
        dq4.metric("Duplicate records",        f"{dup_pct:.1%}")
        date_min_v = df["start_date"].min()
        date_max_v = df["end_date"].max()
        st.caption(f"Date range: **{date_min_v.date() if pd.notna(date_min_v) else 'N/A'}** → **{date_max_v.date() if pd.notna(date_max_v) else 'N/A'}**")

    with st.expander("Methodology Notes", expanded=True):
        if bal_baseline_min is not None:
            st.info(f"**Low-balance filter active**: segments with avg starting balance < €{bal_baseline_min:,.0f} excluded from ranking.")

        st.markdown("**Control group coverage per communication:**")
        ctrl_cov = (
            df.groupby("communication")["contact_flag"]
            .apply(lambda x: (x == 0).mean())
            .reindex(COMM_ORDER).dropna().rename("Control %").to_frame()
        )
        ctrl_cov["Contact %"] = (1 - ctrl_cov["Control %"]).map("{:.1%}".format)
        ctrl_cov["Control %"] = ctrl_cov["Control %"].map("{:.1%}".format)
        st.dataframe(ctrl_cov, use_container_width=True)

        if "tbl" in dir() and not tbl.empty:
            n_cols_tbl = [c for c in tbl.columns if c.endswith("_n")]
            if n_cols_tbl:
                low_n_count  = (tbl[n_cols_tbl] < 30).sum().sum()
                total_cells  = tbl[n_cols_tbl].notna().sum().sum()
                pct_low      = low_n_count / total_cells if total_cells else 0
                msg = f"**Low-confidence cells (N < 30):** {low_n_count:,} of {total_cells:,} ({pct_low:.0%})."
                (st.warning if pct_low > 0.3 else st.info)(msg)

        overlap     = df.groupby("alpha_key")["nsegment"].nunique()
        multi_pct   = (overlap > 1).mean()
        avg_segs_u  = overlap.mean()
        st.info(f"**Segment overlap:** {multi_pct:.1%} of customers in >1 segment (avg {avg_segs_u:.1f}/customer).")

    with st.expander("Top Segment Recommendations", expanded=True):
        if "tbl" not in dir() or tbl.empty:
            st.warning("No table data — adjust filters first.")
        else:
            def _top_tbl(metric_col: str, label: str) -> None:
                if metric_col not in tbl.columns:
                    return
                n_col = next((f"{c}_n" for c in ordered_comms if metric_col.startswith(c + "_")), None)
                cols    = [metric_col] + ([n_col] if n_col and n_col in tbl.columns else [])
                top     = tbl[cols].dropna(subset=[metric_col]).sort_values(metric_col, ascending=False).head(10).copy()
                top.index = [f"{x}  —  {SEGMENT_LABELS.get(str(x), '')}" for x in top.index]
                if n_col and n_col in top.columns:
                    top["Confidence"] = top[n_col].map(
                        lambda n: "🟢 High" if n >= 100 else ("🟡 Medium" if n >= 30 else "🔴 Low"))
                    top[n_col] = top[n_col].map("{:,.0f}".format)
                top[metric_col] = top[metric_col].map("{:.2%}".format)
                top.columns = ([label, "N", "Confidence"] if len(top.columns) == 3 else [label])
                st.dataframe(top, use_container_width=True)

            _fc = ordered_comms[0]  if ordered_comms else None
            _lc = ordered_comms[-1] if ordered_comms else None
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.markdown(f"**Top 10 — {_fc} Bal%**")
                if _fc: _top_tbl(f"{_fc}_bal", "Bal% Δ")
            with rc2:
                st.markdown(f"**Top 10 — {_lc} Bal%**")
                if _lc: _top_tbl(f"{_lc}_bal", "Bal% Δ")
            with rc3:
                st.markdown(f"**Top 10 — {_fc} Lift Bal**")
                if _fc: _top_tbl(f"{_fc}_lift_bal", "Lift Bal")
            with rc4:
                st.markdown(f"**Top 10 — {_fc} Lift Acct**")
                if _fc: _top_tbl(f"{_fc}_lift_acct", "Lift Acct")
