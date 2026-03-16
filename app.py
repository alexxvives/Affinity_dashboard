# app.py — Segment Effect Explorer (v2)
# Run: streamlit run app.py

import ast
import io
from typing import Dict, List, Optional, Tuple

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
    _sdesc_df = pd.read_csv("segment_descriptions.csv", dtype=str)
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
COMM_ORDER = ["day1", "day5", "day7", "day31", "day61", "day90", "day120"]
REQUIRED_COLS = [
    "Communication", "alpha_key", "Contact_flag",
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
        return [s.strip(chr(39)).strip(chr(34))]
    return [str(x)]


@st.cache_data(show_spinner=False)
def _load_csv(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, date_min: Optional[str], date_max: Optional[str], recency_decay: float, bal_clip_pct: float = 0.0) -> pd.DataFrame:
    df = df_raw.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df["Communication"] = df["Communication"].astype(str).str.strip()
    df["Contact_flag"]  = pd.to_numeric(df["Contact_flag"], errors="coerce").fillna(0).astype(int)
    # Use explicit control_flag column from data if present; otherwise infer from Contact_flag
    if "control_flag" in df.columns:
        df["control_flag"] = pd.to_numeric(df["control_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["control_flag"] = (df["Contact_flag"] == 0).astype(int)
    for col in ["start_balance", "end_balance", "start_accounts", "end_accounts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date range filter
    if date_min:
        df = df[df["start_date"] >= pd.Timestamp(date_min)]
    if date_max:
        df = df[df["start_date"] <= pd.Timestamp(date_max)]

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
        return -1


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
@st.cache_data(show_spinner=False)
def agg_cols_data(
    df: pd.DataFrame,
    selected: Tuple[str, ...],
    all_mode: bool,
    bal_min: float,
    acct_min: float,
    bal_baseline_min: Optional[float],
) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["nsegment", "agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci", "agg_lift_bal", "agg_lift_bal_ci", "agg_lift_acct", "agg_lift_acct_ci", "agg_n"])
    if not selected:
        return EMPTY
    treated = df[(df["Contact_flag"] == 1) & (df["Communication"].isin(list(selected)))].copy()
    control = df[(df["control_flag"] == 1) & (df["Communication"].isin(list(selected)))].copy()
    if treated.empty:
        return EMPTY
    treated["_r"] = treated["Communication"].map(_rank)
    if all_mode:
        cnt = (treated.groupby(["alpha_key", "nsegment"])["Communication"]
               .nunique().reset_index(name="_c"))
        ok = cnt[cnt["_c"] >= len(set(selected))][["alpha_key", "nsegment"]]
        treated = treated.merge(ok, on=["alpha_key", "nsegment"], how="inner")
        if treated.empty:
            return EMPTY
    treated = (treated.sort_values("_r", ascending=False)
                      .drop_duplicates(subset=["alpha_key", "nsegment"], keep="first"))
    treated = treated[treated["balance_pct_change"].fillna(-np.inf)  >= bal_min]
    treated = treated[treated["accounts_pct_change"].fillna(-np.inf) >= acct_min]
    # Low-balance filter: exclude segments with mean start_balance below minimum
    if bal_baseline_min is not None:
        seg_base = treated.groupby("nsegment")["start_balance"].mean()
        ok_segs = seg_base[seg_base >= bal_baseline_min].index
        treated = treated[treated["nsegment"].isin(ok_segs)]
    if treated.empty:
        return EMPTY

    rows = []
    for seg, grp in treated.groupby("nsegment", dropna=False):
        w = grp["_weight"]
        n = grp["alpha_key"].nunique()
        mb  = _wmean(grp["balance_pct_change"], w)
        cib = _ci95(grp["balance_pct_change"], w)
        ma  = _wmean(grp["accounts_pct_change"], w)
        cia = _ci95(grp["accounts_pct_change"], w)

        ctrl_grp = control[control["nsegment"] == seg]
        cb  = _wmean(ctrl_grp["balance_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        ca  = _wmean(ctrl_grp["accounts_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        lift_b = (mb - cb) if not np.isnan(mb) and not np.isnan(cb) else np.nan
        lift_a = (ma - ca) if not np.isnan(ma) and not np.isnan(ca) else np.nan
        se_b  = _se(grp["balance_pct_change"], w)
        se_cb = _se(ctrl_grp["balance_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        se_a  = _se(grp["accounts_pct_change"], w)
        se_ca = _se(ctrl_grp["accounts_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        lift_b_ci = float(Z95 * np.sqrt(se_b**2 + se_cb**2)) if not (np.isnan(se_b) or np.isnan(se_cb)) else np.nan
        lift_a_ci = float(Z95 * np.sqrt(se_a**2 + se_ca**2)) if not (np.isnan(se_a) or np.isnan(se_ca)) else np.nan

        rows.append({"nsegment": seg, "agg_bal": mb, "agg_bal_ci": cib,
                     "agg_acct": ma, "agg_acct_ci": cia,
                     "agg_lift_bal": lift_b, "agg_lift_bal_ci": lift_b_ci,
                     "agg_lift_acct": lift_a, "agg_lift_acct_ci": lift_a_ci, "agg_n": n})
    return pd.DataFrame(rows) if rows else EMPTY


@st.cache_data(show_spinner=False)
def comm_data(df: pd.DataFrame, comm: str, bal_min: float, acct_min: float) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["nsegment", f"{comm}_bal", f"{comm}_bal_ci", f"{comm}_acct", f"{comm}_acct_ci", f"{comm}_n", f"{comm}_lift_bal", f"{comm}_lift_bal_ci", f"{comm}_lift_acct", f"{comm}_lift_acct_ci"])
    treated = df[(df["Contact_flag"] == 1) & (df["Communication"] == comm)].copy()
    control = df[(df["control_flag"] == 1) & (df["Communication"] == comm)].copy()
    treated = treated[treated["balance_pct_change"].fillna(-np.inf)  >= bal_min]
    treated = treated[treated["accounts_pct_change"].fillna(-np.inf) >= acct_min]
    if treated.empty:
        return EMPTY
    rows = []
    for seg, grp in treated.groupby("nsegment", dropna=False):
        w = grp["_weight"]
        n = grp["alpha_key"].nunique()
        mb  = _wmean(grp["balance_pct_change"], w)
        cib = _ci95(grp["balance_pct_change"], w)
        ma  = _wmean(grp["accounts_pct_change"], w)
        cia = _ci95(grp["accounts_pct_change"], w)
        ctrl_grp = control[control["nsegment"] == seg]
        cb = _wmean(ctrl_grp["balance_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        ca = _wmean(ctrl_grp["accounts_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        lift_b = (mb - cb) if not (np.isnan(mb) or np.isnan(cb)) else np.nan
        lift_a = (ma - ca) if not (np.isnan(ma) or np.isnan(ca)) else np.nan
        se_b  = _se(grp["balance_pct_change"], w)
        se_cb = _se(ctrl_grp["balance_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        se_a  = _se(grp["accounts_pct_change"], w)
        se_ca = _se(ctrl_grp["accounts_pct_change"], ctrl_grp["_weight"]) if not ctrl_grp.empty else np.nan
        lift_b_ci = float(Z95 * np.sqrt(se_b**2 + se_cb**2)) if not (np.isnan(se_b) or np.isnan(se_cb)) else np.nan
        lift_a_ci = float(Z95 * np.sqrt(se_a**2 + se_ca**2)) if not (np.isnan(se_a) or np.isnan(se_ca)) else np.nan
        rows.append({"nsegment": seg, f"{comm}_bal": mb, f"{comm}_bal_ci": cib,
                     f"{comm}_acct": ma, f"{comm}_acct_ci": cia, f"{comm}_n": n,
                     f"{comm}_lift_bal": lift_b, f"{comm}_lift_bal_ci": lift_b_ci,
                     f"{comm}_lift_acct": lift_a, f"{comm}_lift_acct_ci": lift_a_ci})
    return pd.DataFrame(rows) if rows else EMPTY


@st.cache_data(show_spinner=False)
def build_table(
    df: pd.DataFrame,
    ordered_comms: List[str],
    all_mode: bool,
    agg_thr: Tuple[float, float],
    comm_thr: Dict[str, Tuple[float, float]],
    min_n: int,
    bal_baseline_min: Optional[float],
) -> pd.DataFrame:
    base = pd.DataFrame({"nsegment": sorted(df["nsegment"].unique())})
    a = agg_cols_data(df, tuple(ordered_comms), all_mode, *agg_thr, bal_baseline_min)
    tbl = base.merge(a, on="nsegment", how="left")
    if "agg_n" in tbl.columns:
        mask = tbl["agg_n"].fillna(0) < min_n
        for col in ["agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci",
                    "agg_lift_bal", "agg_lift_bal_ci", "agg_lift_acct", "agg_lift_acct_ci"]:
            if col in tbl.columns:
                tbl.loc[mask, col] = np.nan
    for comm in ordered_comms:
        ca = comm_data(df, comm, *comm_thr.get(comm, (-1.0, -1.0)))
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


def _rdylgn(val: float, lo: float = -0.5, hi: float = 0.5) -> str:
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
        rename[f"{c}_lift_bal_ci"]  = f"{c} \u00b1CI Lift Bal"
        rename[f"{c}_lift_acct"]    = f"{c} Lift Acct"
        rename[f"{c}_lift_acct_ci"] = f"{c} \u00b1CI Lift Acct"
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
        return f"{pct:.1f}%" if abs(pct) < 1.0 else f"{pct:.0f}%"

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

    plain_colour_cols = [v for k, v in rename.items()
                         if (k.endswith("_bal") or k.endswith("_acct")
                             or k in ("agg_bal", "agg_acct"))
                         and "_lift" not in k and "_ci" not in k
                         and v in disp.columns]

    styler = disp.style.format(fmt, na_rep="")
    if plain_colour_cols:
        styler = styler.apply(_colour_col, subset=plain_colour_cols, axis=0)
    # Significance-aware coloring for lift columns
    def _make_lift_styler(lv_c, ci_c):
        def _f(sub_df):
            out = pd.DataFrame("", index=sub_df.index, columns=sub_df.columns)
            for idx in sub_df.index:
                lv = sub_df.loc[idx, lv_c]
                cv = sub_df.loc[idx, ci_c]
                if pd.isna(lv):
                    s_lv = ""
                    s_ci = ""
                elif pd.isna(cv):
                    s_lv = _rdylgn(lv)
                    s_ci = ""
                else:
                    is_sig = (lv - cv) > 0 or (lv + cv) < 0
                    base   = _rdylgn(lv)
                    s_lv   = base if is_sig else base.replace("0.55", "0.20")
                    s_ci   = ("background-color: rgba(80,160,80,0.50); color:#111"
                               if is_sig else
                               "background-color: rgba(200,180,60,0.40); color:#111")
                out.loc[idx, lv_c] = s_lv
                out.loc[idx, ci_c] = s_ci
            return out
        return _f
    for lv_col, ci_col in lift_ci_map.items():
        styler = styler.apply(_make_lift_styler(lv_col, ci_col),
                              subset=[lv_col, ci_col], axis=None)
    # Lift value columns without a matching CI column (plain coloring)
    plain_lift_cols = [v for v in lift_val_cols if v not in lift_ci_map]
    if plain_lift_cols:
        styler = styler.apply(_colour_col, subset=plain_lift_cols, axis=0)
    if n_cols:
        styler = styler.apply(lambda s: s.map(_n_color), subset=n_cols, axis=0)

    table_html = styler.to_html()

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
  tbody td {{ padding: 4px 10px; border-bottom: 1px solid #333; white-space: nowrap; }}
  tbody tr:hover td {{ outline: 1px solid #666; }}
  th.row_heading {{ background: #1c1e2a !important; font-size: 11px;
                    color: #aaa !important; font-weight: normal; cursor: help; }}
  th.blank {{ background: #262730 !important; }}
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
  tbody td {{ padding: 4px 10px; border-bottom: 1px solid #333; white-space: nowrap; }}
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
        out = tbl.reset_index()
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
            col_letter = chr(ord("A") + ci)
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
    txb2.text_frame.text = "Top 20 Segments — Balance % Change (agg)"
    txb2.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
    txb2.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    rows_n = len(top20) + 1
    cols_n = len(key_cols) + 1
    tbl_shape = sl2.shapes.add_table(rows_n, cols_n, Inches(0.3), Inches(0.9), Inches(12.7), Inches(6.2))
    ptbl = tbl_shape.table

    hdr_fill = RGBColor(0x26, 0x27, 0x30)
    headers = ["Segment"] + [c.replace("_", " ").title() for c in key_cols]
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

# ── Load raw CSV ──────────────────────────────────────────────────────────────
try:
    with open("dummy_segment_data.csv", "rb") as f:
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
    st.subheader("Display")
    # Square compact CSS for the flanking ±1 buttons
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stButton button {
        min-width: 1.3rem !important; max-width: 1.3rem !important;
        height: 1.3rem !important; padding: 0 !important;
        font-size: 0.7rem !important; line-height: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # ── min_n: slider + −1/+1 flanking buttons ───────────────────────────────
    if "min_n_val" not in st.session_state:
        st.session_state["min_n_val"] = 0
    # Clamp stored value if dataset changed
    st.session_state["min_n_val"] = max(0, min(_max_slider_n, st.session_state["min_n_val"]))
    st.caption("Hide cells with fewer than N users")
    _mn_c1, _mn_c2, _mn_c3 = st.columns([1, 10, 1])
    if _mn_c1.button("−", key="mn_m1"):
        st.session_state["min_n_val"] = max(0, st.session_state["min_n_val"] - 1)
    if _mn_c3.button("+", key="mn_p1"):
        st.session_state["min_n_val"] = min(_max_slider_n, st.session_state["min_n_val"] + 1)
    with _mn_c2:
        min_n = st.slider("N", 0, _max_slider_n, key="min_n_val", label_visibility="collapsed")

    st.divider()
    with st.expander("⚙️ Advanced filters", expanded=False):
        all_mode = st.toggle(
            "Only count customers who received all selected communications",
            value=False,
            help="When ON, a customer is only counted if they appear in every selected communication step.",
        )
        agg_thr = (-1.0, -1.0)
        comm_thr: Dict[str, Tuple[float, float]] = {c: agg_thr for c in COMM_ORDER}
        st.markdown("---")
        bal_clip_pct = st.slider(
            "Clip extreme balance % changes (percentile)",
            min_value=0, max_value=20, value=5, step=1,
            help="Removes the most extreme values at both ends. 5 = clip below 5th and above 95th percentile.",
        )
        st.markdown("---")
        st.markdown("**⚠️ Exclude low-balance segments**")
        st.caption(
            "Segments where customers had very little money at the start can show big % gains "
            "from a small euro increase. Set a minimum to remove them from the ranking."
        )
        bal_baseline_cap_raw = st.slider(
            "Min. avg customer starting balance (€, 0 = off)",
            min_value=0, max_value=10000, value=0, step=250,
        )
        bal_baseline_min = float(bal_baseline_cap_raw) if bal_baseline_cap_raw > 0 else None

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
_all_present_comms = [c for c in COMM_ORDER if c in df["Communication"].unique()]
ordered_comms = [c for c in _all_present_comms if st.session_state.get(f"cb_{c}", True)]
if not ordered_comms:
    ordered_comms = _all_present_comms[:]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_explorer, tab_table, tab_simulator, tab_charts, tab_export, tab_audit = st.tabs([
    "Segment Explorer",
    "Table",
    "Targeting Simulator",
    "Charts",
    "Export",
    "Audit",
])


# ══════════════════════════════════════════════════════════
# SEGMENT EXPLORER TAB
# ══════════════════════════════════════════════════════════
with tab_explorer:
    st.subheader("Segment catalogue")
    st.caption("Browse all segments by category. Unique Users = contacted users in the active date range.")
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
            df[df["Contact_flag"] == 1]
            .groupby("nsegment")["alpha_key"]
            .nunique()
            .rename("Unique Users")
            .reset_index()
            .rename(columns={"nsegment": "Segment ID"})
        )
        _exp_base = _exp_base.merge(_user_counts, on="Segment ID", how="left")
        _exp_base["Unique Users"] = _exp_base["Unique Users"].fillna(0).astype(int)
        _exp_base = _exp_base.sort_values(["Group", "Segment ID"]).reset_index(drop=True)

        # ── Group filter ────────────────────────────────────────────────────────
        _all_groups = sorted(_exp_base["Group"].unique().tolist())
        _sel_groups = st.multiselect(
            "Filter by group", _all_groups, default=_all_groups, key="explorer_groups"
        )
        _filtered_exp = _exp_base[_exp_base["Group"].isin(_sel_groups)].copy()

        # ── Full segment table (always visible) ────────────────────────────────
        st.dataframe(_filtered_exp, use_container_width=True, hide_index=True)
        st.caption(f"{len(_filtered_exp):,} segments — {len(_sel_groups)} group(s) selected.")

        st.divider()

        # ── Group size chart ───────────────────────────────────────────────────────
        _grp_sz = (
            _filtered_exp.groupby("Group")
            .agg(Segments=("Segment ID", "count"), Users=("Unique Users", "sum"))
            .reset_index()
            .sort_values("Users", ascending=False)
        )
        _ec1, _ec2 = st.columns(2)
        with _ec1:
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
        with _ec2:
            _fig_gus = px.bar(
                _grp_sz, x="Group", y="Users",
                color="Users", color_continuous_scale="Teal",
                text=_grp_sz["Users"].map(lambda v: f"{v:,}"),
                title="Unique users per group (active date range)",
            )
            _fig_gus.update_xaxes(tickangle=-30)
            _fig_gus.update_traces(textposition="outside")
            _fig_gus.update_layout(
                coloraxis_showscale=False, height=360,
                yaxis_range=[0, int(_grp_sz["Users"].max()) * 1.20],
            )
            st.plotly_chart(_fig_gus, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TABLE TAB
# ══════════════════════════════════════════════════════════
with tab_table:
    # ── Row 1: comm toggle strip + show lift at right ─────────────────────────
    _n_pc = len(_all_present_comms)
    _row1 = st.columns(_n_pc + 1)
    for _i, _c in enumerate(_all_present_comms):
        _row1[_i].checkbox(_c, value=True, key=f"cb_{_c}")
    show_lift = _row1[_n_pc].checkbox("Show lift vs control", value=False, key="tbl_show_lift")
    ordered_comms = [c for c in _all_present_comms if st.session_state.get(f"cb_{c}", True)]
    if not ordered_comms:
        ordered_comms = _all_present_comms[:]

    # ── Row 2: show columns multiselect (Balance / Accounts / Sample size) ─────
    _metric_sel = st.multiselect(
        "Show columns",
        ["Balance", "Accounts", "Sample size"],
        default=["Balance", "Accounts"],
        key="show_metric_ms",
    )
    _sel = _metric_sel or ["Balance", "Accounts"]
    _show_bal  = "Balance"     in _sel
    _show_acct = "Accounts"    in _sel
    show_n_cols = "Sample size" in _sel
    if _show_bal and not _show_acct:
        _show_metric_val = "balance"
    elif _show_acct and not _show_bal:
        _show_metric_val = "accounts"
    else:
        _show_metric_val = "both"

    with st.spinner("Computing..."):
        tbl = build_table(df, ordered_comms, all_mode, agg_thr, comm_thr, min_n, bal_baseline_min)

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
        components.html(html_tbl, height=680, scrolling=True)
        st.caption(
            "All % changes are measured over the 7-day window between start_date and end_date. "
            "Click any column header to sort. "
            "Hover over a segment ID for its description."
        )
        if show_lift:
            st.caption(
                "🟢 **Strong color** = lift 95% CI does not cross zero (statistically significant).  "
                "🟡 **Muted color** = CI crosses zero (direction not reliable at 95% level).  "
                "The **\u00b1CI Lift** column shows the half-width of the 95% confidence interval."
            )


# ══════════════════════════════════════════════════════════
# CHARTS TAB
# ══════════════════════════════════════════════════════════
with tab_charts:
    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Table tab.")
    else:
        st.caption(
            "**Top segments bar chart** — ranks segments by the selected metric. "
            "Taller bars = stronger average effect for that communication. "
            "Use this to decide which segments to prioritise in the next campaign wave. "
            "Error bars (where shown) are 95% confidence intervals — wider bars mean less certainty."
        )
        # ── Bar chart ────────────────────────────────────────────────────────
        _TOP_N_CHART = 25
        metric_opts = ([f"{c} Bal%"       for c in ordered_comms]
                       + [f"{c} Acct%"      for c in ordered_comms]
                       + [f"{c} Lift Bal"   for c in ordered_comms]
                       + [f"{c} Lift Acct"  for c in ordered_comms])
        chosen = st.selectbox("Sort / highlight metric", metric_opts, key="chart_cm")

        col_map = {
            **{f"{c} Bal%":      f"{c}_bal"       for c in ordered_comms},
            **{f"{c} Acct%":     f"{c}_acct"      for c in ordered_comms},
            **{f"{c} Lift Bal":  f"{c}_lift_bal"  for c in ordered_comms},
            **{f"{c} Lift Acct": f"{c}_lift_acct" for c in ordered_comms},
        }
        _default_sort = f"{ordered_comms[0]}_bal" if ordered_comms else ""
        sort_col = col_map.get(chosen, _default_sort)

        if sort_col in tbl.columns:
            top_data = tbl[sort_col].dropna().sort_values(ascending=False).head(_TOP_N_CHART)
            if "_lift_" in sort_col:
                ci_col = None
            else:
                ci_col = sort_col.replace("_bal", "_bal_ci").replace("_acct", "_acct_ci")
            has_ci = ci_col is not None and ci_col in tbl.columns

            x_labels = [str(x) for x in top_data.index]
            _bar_hover = [
                (SEGMENT_LABELS.get(x, "") + (" — " + SEGMENT_DESCRIPTIONS.get(x, "") if SEGMENT_DESCRIPTIONS.get(x) else ""))
                for x in x_labels
            ]
            fig_bar = px.bar(
                x=x_labels,
                y=top_data.values * 100,
                error_y=(tbl.loc[top_data.index, ci_col].values * 100) if has_ci else None,
                color=top_data.values * 100,
                color_continuous_scale="RdYlGn",
                labels={"x": "Segment", "y": "Mean % change", "color": "%"},
                title=f"Top {_TOP_N_CHART} segments — {chosen}",
            )
            fig_bar.update_traces(
                customdata=_bar_hover,
                hovertemplate="<b>%{x}</b>: %{y:.1f}%<br><span style='color:#aaa'>%{customdata}</span><extra></extra>",
            )
            fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45, height=480)
            fig_bar.update_xaxes(
                type="category",
                tickvals=x_labels,
                ticktext=[f"{x}  ·  {SEGMENT_LABELS[x]}" if SEGMENT_LABELS.get(x) else x for x in x_labels],
            )
            fig_bar.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Selected metric not available for current communication selection.")

        st.divider()

        # ── Heatmap ─────────────────────────────────────────────────────────
        _hm_metric = st.radio("Heatmap metric", ["Balance %", "Accounts %"], horizontal=True, key="hm_metric")
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
                ticktext=[f"{s}  ·  {SEGMENT_LABELS[s]}" if SEGMENT_LABELS.get(s) else str(s) for s in hm.index],
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
            format_func=lambda x: f"{x}  —  {SEGMENT_LABELS.get(x, x)}",
            key="jt_segs",
        )
        if jt_segs:
            jt_rows = []
            for comm in ordered_comms:
                sub = df[(df["Contact_flag"] == 1) & (df["Communication"] == comm) & (df["nsegment"].isin(jt_segs))]
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
            "**Co-occurrence heatmap** — cell [A, B] = fraction of segment A users "
            "who are also in segment B. Dark blue = heavy overlap. "
            "If two high-performing segments overlap heavily, targeting both wastes budget — "
            "pick the one with stronger lift. Also useful for building exclusion lists."
        )
        _cooc_sort = next((f"{c}_bal" for c in ordered_comms if f"{c}_bal" in tbl.columns), None)
        top_segs_cooc = (tbl.sort_values(_cooc_sort, ascending=False).head(30).index.astype(str).tolist()
                         if _cooc_sort else tbl.index.astype(str).tolist()[:30])
        if len(top_segs_cooc) < 2:
            st.info("Not enough segments with data to compute co-occurrence. Adjust filters.")
        else:
            _cooc_max = len(top_segs_cooc)
            if _cooc_max <= 2:
                cooc_n = _cooc_max
            else:
                _cooc_def = min(20, _cooc_max)
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
            cooc_matrix = pd.DataFrame(0.0, index=cooc_segs, columns=cooc_segs)
            for _, row in user_seg_df.iterrows():
                segs = list(row["nsegment"])
                for i, si in enumerate(segs):
                    for sj in segs:
                        if si in cooc_matrix.index and sj in cooc_matrix.columns:
                            cooc_matrix.loc[si, sj] += 1
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
                labels=dict(color="Share of row-seg users"),
                title="Segment co-occurrence (row = % of row-segment users who also belong to col-segment)",
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

        # ── Violin / distribution ────────────────────────────────────────────
        _vio_metric = st.radio("Violin metric", ["Balance %", "Accounts %"], horizontal=True, key="vio_metric")
        _vio_col    = "balance_pct_change" if _vio_metric == "Balance %" else "accounts_pct_change"
        _vio_label  = "Balance % change"   if _vio_metric == "Balance %" else "Accounts % change"
        st.subheader(f"{_vio_label} distribution by communication")
        st.caption(
            "**Violin chart** — shows the full spread of individual % changes, "
            "not just the average. A wide violin = high variability, meaning the average is driven by "
            "a few extreme responders. A narrow violin = consistent response across all customers. "
            "Use this to identify segments where the average is misleading."
        )
        pool     = tbl.index.astype(str).tolist()[:30]
        seg_pick = st.multiselect(
            "Segments to compare (top 30)",
            options=pool, default=pool[:4],
            format_func=lambda x: f"{x}  —  {SEGMENT_LABELS.get(x, x)}",
            key="chart_vio",
        )
        if seg_pick:
            dv = df[
                (df["Contact_flag"] == 1) & (df["nsegment"].isin(seg_pick))
                & (df["Communication"].isin(ordered_comms))
            ].dropna(subset=[_vio_col])
            if not dv.empty:
                dv = dv.copy()
                dv["_label"] = dv["nsegment"].astype(str)
                fig_v = px.violin(
                    dv, x="Communication", y=_vio_col,
                    color="_label", box=True, points=False,
                    category_orders={"Communication": ordered_comms},
                    labels={_vio_col: _vio_label, "_label": "Segment"},
                    title=f"{_vio_label} distribution per communication",
                )
                fig_v.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_v, use_container_width=True)

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
                                       format_func=lambda x: f"{x}  —  {SEGMENT_LABELS.get(x, x)}", key="dist_segs")
        with dc2:
            dist_comm = st.selectbox("Communication", options=["All selected"] + ordered_comms, key="dist_comm")
        with dc3:
            dist_metric = st.radio("Metric", ["Balance %", "Accounts %"], horizontal=True, key="dist_metric")

        if dist_segs:
            dm = df[(df["Contact_flag"] == 1) & (df["nsegment"].isin(dist_segs))]
            if dist_comm != "All selected":
                dm = dm[dm["Communication"] == dist_comm]
            else:
                dm = dm[dm["Communication"].isin(ordered_comms)]
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
# TARGETING SIMULATOR TAB
# ══════════════════════════════════════════════════════════
with tab_simulator:
    st.subheader("🎯 Targeting Simulator")
    st.caption(
        "Select any set of segments below and we'll estimate the **expected balance lift per "
        "communication** if you sent to only those customers. "
        "Lift is N-weighted across your chosen segments: segments with more users carry more weight."
    )

    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Table tab first.")
    else:
        all_segs_sim = sorted(tbl.index.astype(str).tolist())
        _default_sim = all_segs_sim[:min(10, len(all_segs_sim))]

        _sc1, _sc2 = st.columns([3, 1])
        with _sc1:
            sim_segs = st.multiselect(
                "Segments to include in the simulation",
                options=all_segs_sim,
                default=_default_sim,
                format_func=lambda x: f"{x}  —  {SEGMENT_LABELS.get(x, x)}" if SEGMENT_LABELS.get(x) else x,
                key="sim_segs",
            )
        with _sc2:
            sim_metric = st.radio(
                "Metric",
                ["Balance % lift", "Accounts % lift"],
                key="sim_metric",
            )
        _lift_suffix = "_lift_bal" if sim_metric == "Balance % lift" else "_lift_acct"
        _raw_suffix  = "_bal"      if sim_metric == "Balance % lift" else "_acct"
        _y_label     = "Expected Balance Lift %" if sim_metric == "Balance % lift" else "Expected Accounts Lift %"

        if sim_segs:
            # ── Pre-filter table rows and compute projected metrics ───────────
            sub_all = tbl.loc[tbl.index.isin(sim_segs)].copy()
            _sim_total_users = df[
                (df["Contact_flag"] == 1) & (df["nsegment"].isin(sim_segs))
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
                    (df["Contact_flag"] == 1)
                    & (df["Communication"] == comm)
                    & (df["nsegment"].isin(sim_segs))
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

            st.metric(
                "Total unique users (all selected segments)",
                f"{_sim_total_users:,}",
            )

            # Show kpi metrics row
            st.markdown("#### Expected lift across selected segments")
            kpi_cols = st.columns(len(sim_summary))
            for i, row in sim_summary.iterrows():
                lift_val = row["Expected Lift"]
                label    = row["Communication"]
                n_val    = int(row["Total N"]) if pd.notna(row["Total N"]) else 0
                delta_str = f"N = {n_val:,}"
                if pd.notna(lift_val):
                    kpi_cols[i].metric(label, f"{lift_val:.2%}", delta_str)
                else:
                    kpi_cols[i].metric(label, "—", delta_str)

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
                        f"{nu:,} users",
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
                        f"{nu:,} users",
                    )

            st.divider()

            # ── Per-segment detail table ──────────────────────────────────────
            st.markdown("#### Per-segment breakdown")
            _c_sim_cap, _c_sim_tog = st.columns([6, 2])
            _c_sim_cap.caption(
                "Each cell shows the lift (treatment \u2212 control) for that segment \u00d7 communication. "
                "Blank = no control data or fewer users than the minimum N filter. "
                "Hover a segment ID for its description."
            )
            _show_sim_n = _c_sim_tog.toggle("Show sample size", value=False, key="sim_show_n")

            # Build columns: lift only, or interleaved lift + N
            _intl_cols = []
            for c in ordered_comms:
                if f"{c}{_lift_suffix}" in tbl.columns:
                    _intl_cols.append(f"{c}{_lift_suffix}")
                if _show_sim_n and f"{c}_n" in tbl.columns:
                    _intl_cols.append(f"{c}_n")
            detail = tbl.loc[tbl.index.isin(sim_segs), [c for c in _intl_cols if c in tbl.columns]].copy()
            detail.index.name = "Segment"

            col_rename_sim = {f"{c}{_lift_suffix}": c for c in ordered_comms}
            col_rename_sim.update({f"{c}_n": f"{c} N" for c in ordered_comms})
            detail = detail.rename(columns=col_rename_sim)

            def _pct_fmt_sim(v):
                if pd.isna(v): return ""
                pct = v * 100
                return f"{pct:.1f}%" if abs(pct) < 1.0 else f"{pct:.0f}%"

            pct_fmt = {c: _pct_fmt_sim for c in ordered_comms if c in detail.columns}
            n_fmt   = {f"{c} N": "{:,.0f}" for c in ordered_comms if f"{c} N" in detail.columns}
            fmt_all = {**pct_fmt, **n_fmt}

            def _lift_color(col_series):
                return col_series.map(_rdylgn)

            def _n_color_sim(col_series):
                return col_series.map(_n_color)

            lift_disp_cols = [c for c in ordered_comms if c in detail.columns]
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
        else:
            st.info("Select at least one segment above to run the simulation.")


# ══════════════════════════════════════════════════════════
# EXPORT TAB
# ══════════════════════════════════════════════════════════
with tab_export:
    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Table tab first.")
    else:
        _n_segs  = len(tbl)
        _n_comms = len(ordered_comms)
        _n_cols  = len(tbl.columns)

        # ── Header ───────────────────────────────────────────────────────────
        st.markdown("## Export data")
        st.caption(
            f"Current view: **{_n_segs}** segments · **{_n_comms}** communications · **{_n_cols}** columns. "
            "Choose a format below."
        )
        st.divider()

        # ── Download cards ────────────────────────────────────────────────────
        _card_css = """
<style>
.exp-card {
    background: #1e2030;
    border: 1px solid #383a52;
    border-radius: 10px;
    padding: 22px 20px 18px 20px;
    height: 100%;
}
.exp-card h3 { margin: 0 0 6px 0; font-size: 15px; color: #e0e0ff; }
.exp-card .badge {
    display: inline-block;
    background: #2e3154;
    color: #a0aecf;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    border-radius: 4px;
    padding: 2px 7px;
    margin-bottom: 10px;
}
.exp-card p { font-size: 12px; color: #8a8fac; line-height: 1.6; margin: 0; }
</style>"""
        components.html(_card_css + "<div></div>", height=0)

        xc1, xc2, xc3 = st.columns(3, gap="medium")

        with xc1:
            st.markdown(
                """<div class="exp-card"><h3>CSV</h3></div>""",
                unsafe_allow_html=True,
            )
            csv_bytes = tbl.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_bytes, "segment_table.csv", "text/csv",
                use_container_width=True,
            )

        with xc2:
            st.markdown(
                """<div class="exp-card"><h3>Excel (.xlsx)</h3></div>""",
                unsafe_allow_html=True,
            )
            try:
                xlsx_bytes = build_excel(tbl, ordered_comms)
                st.download_button(
                    "Download Excel",
                    xlsx_bytes, "segment_table.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Excel error: {e}")

        with xc3:
            st.markdown(
                """<div class="exp-card"><h3>PowerPoint (.pptx)</h3></div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            try:
                pptx_bytes = build_pptx(tbl, ordered_comms)
                st.download_button(
                    "Download PowerPoint",
                    pptx_bytes, "segment_report.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PowerPoint error: {e}")

        st.divider()

        # ── Data preview ─────────────────────────────────────────────────────
        with st.expander("Preview data", expanded=True):
            _prev = tbl.reset_index()
            _prev_disp = _prev.head(10)
            _pct_cols_prev = [c for c in _prev_disp.columns if any(c.endswith(s) for s in ("_bal", "_acct", "_lift_bal", "_lift_acct"))]
            st.dataframe(
                _prev_disp.style.format(
                    {c: "{:.1%}" for c in _pct_cols_prev if c in _prev_disp.columns},
                    na_rep="—"
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing first 10 of {_n_segs} segments.")


# ══════════════════════════════════════════════════════════
# AUDIT TAB
# ══════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("Tool Audit & Recommendations")
    st.caption("Live analysis of the current dataset and active filter settings.")

    k1, k2, k3, k4 = st.columns(4)
    n_rows_raw   = len(df_raw)
    n_users_raw  = df_raw["alpha_key"].nunique() if "alpha_key" in df_raw.columns else 0
    n_segs       = df["nsegment"].nunique()
    contact_rate = df_raw["Contact_flag"].mean() if "Contact_flag" in df_raw.columns else 0
    k1.metric("Total rows",      f"{n_rows_raw:,}")
    k2.metric("Unique users",    f"{n_users_raw:,}")
    k3.metric("Unique segments", f"{n_segs:,}")
    k4.metric("Contact rate",    f"{contact_rate:.1%}")

    st.divider()

    with st.expander("📋 Data Quality", expanded=True):
        nan_bal  = df["balance_pct_change"].isna().mean()
        nan_acct = df["accounts_pct_change"].isna().mean()
        zero_bal = (df["start_balance"].fillna(0) < 1).mean()
        dup_pct  = df.duplicated(subset=["alpha_key", "Communication", "nsegment"]).mean()
        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Balance % NaN rate",     f"{nan_bal:.1%}")
        dq2.metric("Accounts % NaN rate",    f"{nan_acct:.1%}")
        dq3.metric("Near-zero balance rows", f"{zero_bal:.1%}")
        dq4.metric("Duplicate rows",         f"{dup_pct:.1%}")
        date_min_v = df["start_date"].min()
        date_max_v = df["end_date"].max()
        st.caption(f"Date range: **{date_min_v.date() if pd.notna(date_min_v) else 'N/A'}** → **{date_max_v.date() if pd.notna(date_max_v) else 'N/A'}**")

    with st.expander("⚠️ Methodology Risks", expanded=True):
        slider_active = (agg_thr[0] > -1.0 or agg_thr[1] > -1.0
                         or any(v[0] > -1.0 or v[1] > -1.0 for v in comm_thr.values()))
        if slider_active:
            st.warning("**Survivorship bias active.** Threshold sliders above −100% inflate results.")
        else:
            st.success("✓ No threshold filters active.")

        if bal_baseline_min is not None:
            st.info(f"**Low-balance filter active**: segments with avg starting balance < €{bal_baseline_min:,.0f} excluded from ranking.")

        st.markdown("**Control group coverage per communication:**")
        ctrl_cov = (
            df.groupby("Communication")["Contact_flag"]
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
        st.info(f"**Segment overlap:** {multi_pct:.1%} of users in >1 segment (avg {avg_segs_u:.1f}/user).")

    with st.expander("🏆 Top Segment Recommendations", expanded=True):
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
