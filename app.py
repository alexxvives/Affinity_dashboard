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
    # control_flag is complement of Contact_flag
    df["control_flag"] = 1 - df["Contact_flag"]
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
    EMPTY = pd.DataFrame(columns=["nsegment", "agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci", "agg_lift_bal", "agg_lift_acct", "agg_n"])
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

        rows.append({"nsegment": seg, "agg_bal": mb, "agg_bal_ci": cib,
                     "agg_acct": ma, "agg_acct_ci": cia,
                     "agg_lift_bal": lift_b, "agg_lift_acct": lift_a, "agg_n": n})
    return pd.DataFrame(rows) if rows else EMPTY


@st.cache_data(show_spinner=False)
def comm_data(df: pd.DataFrame, comm: str, bal_min: float, acct_min: float) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["nsegment", f"{comm}_bal", f"{comm}_bal_ci", f"{comm}_acct", f"{comm}_acct_ci", f"{comm}_n", f"{comm}_lift_bal", f"{comm}_lift_acct"])
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
        rows.append({"nsegment": seg, f"{comm}_bal": mb, f"{comm}_bal_ci": cib,
                     f"{comm}_acct": ma, f"{comm}_acct_ci": cia, f"{comm}_n": n,
                     f"{comm}_lift_bal": lift_b, f"{comm}_lift_acct": lift_a})
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
        for col in ["agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci", "agg_lift_bal", "agg_lift_acct"]:
            if col in tbl.columns:
                tbl.loc[mask, col] = np.nan
    for comm in ordered_comms:
        ca = comm_data(df, comm, *comm_thr.get(comm, (-1.0, -1.0)))
        tbl = tbl.merge(ca, on="nsegment", how="left")
        nc = f"{comm}_n"
        if nc in tbl.columns:
            mask = tbl[nc].fillna(0) < min_n
            for col in [f"{comm}_bal", f"{comm}_bal_ci", f"{comm}_acct", f"{comm}_acct_ci", f"{comm}_lift_bal", f"{comm}_lift_acct"]:
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
        "agg_bal":       "Bal%\u0394 (agg)",
        "agg_bal_ci":    "\u00b1CI Bal (agg)",
        "agg_acct":      "Acct%\u0394 (agg)",
        "agg_acct_ci":   "\u00b1CI Acct (agg)",
        "agg_lift_bal":  "Lift Bal (agg)",
        "agg_lift_acct": "Lift Acct (agg)",
        "agg_n":         "N (agg)",
    }
    for c in ordered_comms:
        rename[f"{c}_bal"]    = f"{c} Bal%"
        rename[f"{c}_bal_ci"] = f"{c} \u00b1CI"
        rename[f"{c}_acct"]   = f"{c} Acct%"
        rename[f"{c}_acct_ci"]= f"{c} \u00b1CI Acct"
        rename[f"{c}_n"]          = f"{c} N"
        rename[f"{c}_lift_bal"]   = f"{c} Lift Bal"
        rename[f"{c}_lift_acct"]  = f"{c} Lift Acct"
    disp = disp.rename(columns=rename)

    # Drop columns based on toggles
    drop_cols = []
    # Always drop aggregate columns from table (still available in tbl for charts)
    _agg_keys = {"agg_bal", "agg_bal_ci", "agg_acct", "agg_acct_ci", "agg_lift_bal", "agg_lift_acct", "agg_n"}
    drop_cols += [v for k, v in rename.items() if k in _agg_keys and v in disp.columns]
    if not show_n_cols:
        drop_cols += [v for k, v in rename.items() if k.endswith("_n") and v in disp.columns]
    # Always drop CI from table (CI data stays in tbl DataFrame for chart error bars)
    drop_cols += [v for k, v in rename.items() if ("_ci" in k) and v in disp.columns]
    # Lift is an exclusive mode: show lift columns OR raw % columns, never both
    if show_lift:
        # drop per-comm raw % columns; keep per-comm lift columns
        drop_cols += [v for k, v in rename.items()
                      if v in disp.columns and "_ci" not in k and "_lift" not in k
                      and (k.endswith("_bal") or k.endswith("_acct"))]
    else:
        # drop per-comm lift columns; show raw % columns
        drop_cols += [v for k, v in rename.items() if ("_lift" in k) and v in disp.columns]
    disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns])

    pct_cols = [v for k, v in rename.items()
                if (k.endswith("_bal") or k.endswith("_acct") or k in ("agg_bal", "agg_acct", "agg_lift_bal", "agg_lift_acct") or "_lift" in k or "_ci" in k)
                and v in disp.columns]
    n_cols = [v for k, v in rename.items() if (k.endswith("_n") or k == "agg_n") and v in disp.columns]

    fmt = {col: "{:.2%}" for col in pct_cols if col in disp.columns}
    fmt.update({col: "{:,.0f}" for col in n_cols if col in disp.columns})

    def _colour_col(col_series: pd.Series) -> pd.Series:
        return col_series.map(_rdylgn)

    colour_cols = [v for k, v in rename.items()
                   if (k.endswith("_bal") or k.endswith("_acct") or k in ("agg_bal", "agg_acct") or "_lift" in k)
                   and v in disp.columns]

    styler = disp.style.format(fmt, na_rep="")
    if colour_cols:
        styler = styler.apply(_colour_col, subset=colour_cols, axis=0)
    if n_cols:
        styler = styler.apply(lambda s: s.map(_n_color), subset=n_cols, axis=0)

    # Build HTML manually to get proper CSS-hover tooltips on the nsegment index
    table_html = styler.to_html()

    # Inject tooltip CSS spans into the index cells using string replacement
    if seg_labels or seg_desc:
        import re
        def _replace_idx_cell(m):
            cell_id = m.group(1).strip()
            label = seg_labels.get(cell_id, "") if seg_labels else ""
            desc  = seg_desc.get(cell_id, "")   if seg_desc  else ""
            tooltip_text = desc if desc else label
            if tooltip_text:
                display = f'<span class="seg-id">{cell_id}</span><span class="seg-tip">{tooltip_text}</span>'
            else:
                display = cell_id
            return f'<th class="row_heading level0">{display}</th>'
        # Flexible regex — handles id="..." attr before or after class, pandas 1.x/2.x both
        table_html = re.sub(
            r'<th[^>]+class="[^"]*row_heading level0[^"]*"[^>]*>\s*([^<]+?)\s*</th>',
            _replace_idx_cell,
            table_html,
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
                    color: #aaa !important; font-weight: normal; position: relative; }}
  th.blank {{ background: #262730 !important; }}
  .seg-id {{ display: inline-block; }}
  .seg-tip {{
    display: none;
    position: absolute;
    left: 110%;
    top: 0;
    z-index: 9999;
    background: #1a1a2e;
    color: #eee;
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 12px;
    max-width: 2100px;
    white-space: normal;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    pointer-events: none;
  }}
  th.row_heading:hover .seg-tip {{ display: block; }}
</style></head>
<body><div style="overflow:auto; max-height: 615px;">
{table_html}
</div></body></html>"""


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
    tf.text = "Nsegment Explorer"
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
st.set_page_config(page_title="Nsegment Explorer", layout="wide")
st.title("Nsegment Explorer")

st.markdown("""
<style>
span[data-baseweb="tag"] { background-color: #31333F !important; }
span[data-baseweb="tag"] span { color: #FAFAFA !important; }
span[data-baseweb="tag"] button { color: #FAFAFA !important; }
div.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    selected_comms = st.multiselect("Communications", options=COMM_ORDER, default=COMM_ORDER)

    st.divider()
    st.subheader("Date range")
    _dates = pd.to_datetime(df_raw["start_date"], errors="coerce").dropna()
    _d_min = _dates.min().date()
    _d_max = _dates.max().date()
    date_from = st.date_input("From", value=_d_min, min_value=_d_min, max_value=_d_max)
    date_to   = st.date_input("To",   value=_d_max, min_value=_d_min, max_value=_d_max)

    st.divider()
    st.subheader("Display")
    show_n_cols = st.checkbox("Show N (sample size) columns", value=False)
    show_lift   = st.checkbox("Show lift vs control (replaces raw %)", value=False)
    min_n       = st.slider("Hide cells with fewer than N users", 0, 100, 3, step=1,
                            help="Cells with fewer unique contacted users are shown as blank.")

    st.divider()
    with st.expander("⚙️ Advanced filters", expanded=False):
        all_mode = st.toggle(
            "Only count customers who received all selected communications",
            value=False,
            help="When ON, a customer is only counted if they appear in every selected communication step.",
        )
        st.caption("Raise these to focus only on customers who showed improvement. Leave at −100% for unbiased results.")
        agg_b_raw = st.slider("Min. balance % change", -100, 200, -100, key="k_ab")
        agg_a_raw = st.slider("Min. accounts % change", -100, 200, -100, key="k_aa")
        agg_thr = (agg_b_raw / 100.0, agg_a_raw / 100.0)
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

ordered_comms = [c for c in COMM_ORDER if c in selected_comms]
if not ordered_comms:
    st.info("Select at least one communication in the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_charts, tab_export, tab_audit = st.tabs([
    "📊  Table",
    "📈  Charts",
    "⬇  Export",
    "🔍  Audit",
])


# ══════════════════════════════════════════════════════════
# TABLE TAB
# ══════════════════════════════════════════════════════════
with tab_table:
    with st.spinner("Computing..."):
        tbl = build_table(df, ordered_comms, all_mode, agg_thr, comm_thr, min_n, bal_baseline_min)

    if tbl.empty:
        st.warning("No segments pass the current filters. Try adjusting filters in the sidebar.")
    else:
        html_tbl = style_tbl(
            tbl, ordered_comms,
            seg_labels=SEGMENT_LABELS or None,
            seg_desc=SEGMENT_DESCRIPTIONS or None,
            show_n_cols=show_n_cols,
            show_lift=show_lift,
        )
        components.html(html_tbl, height=680, scrolling=True)


# ══════════════════════════════════════════════════════════
# CHARTS TAB
# ══════════════════════════════════════════════════════════
with tab_charts:
    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Table tab.")
    else:
        st.caption(
            "📊 **Top segments bar chart** — ranks segments by the selected metric. "
            "Taller bars = stronger average effect for that communication. "
            "Use this to decide which segments to prioritise in the next campaign wave. "
            "Error bars (where shown) are 95% confidence intervals — wider bars mean less certainty."
        )
        # ── Bar chart ────────────────────────────────────────────────────────
        ctrl1, ctrl2 = st.columns([1, 3])
        with ctrl1:
            top_n = st.slider("Top N segments", 5, min(150, len(tbl)), min(30, len(tbl)), key="chart_tn")
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
            top_data = tbl[sort_col].dropna().sort_values(ascending=False).head(top_n)
            if "_lift_" in sort_col:
                ci_col = None  # lift columns have no CI band
            else:
                ci_col = sort_col.replace("_bal", "_bal_ci").replace("_acct", "_acct_ci")
            has_ci = ci_col is not None and ci_col in tbl.columns

            x_labels = [f"{x}  {SEGMENT_LABELS.get(str(x), '')}" for x in top_data.index]
            fig_bar = px.bar(
                x=x_labels,
                y=top_data.values * 100,
                error_y=(tbl.loc[top_data.index, ci_col].values * 100) if has_ci else None,
                color=top_data.values * 100,
                color_continuous_scale="RdYlGn",
                labels={"x": "Segment", "y": "Mean % change", "color": "%"},
                title=f"Top {top_n} segments — {chosen}",
            )
            fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45, height=420)
            fig_bar.update_yaxes(ticksuffix="%")
            with ctrl2:
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            with ctrl2:
                st.info("Selected metric not available for current communication selection.")

        st.divider()

        # ── Heatmap ─────────────────────────────────────────────────────────
        st.subheader("Balance % change — segment × communication")
        st.caption(
            "🟩 **Heatmap** — each cell is the avg balance % change for a segment (row) "
            "at a single communication touchpoint (column). Darker green = stronger positive response. "
            "Use this to spot which communications resonate with which segments, and identify "
            "segments that respond early vs late in the journey."
        )
        heat_cols = [f"{c}_bal" for c in ordered_comms if f"{c}_bal" in tbl.columns]
        if heat_cols:
            hm = tbl[heat_cols].copy()
            hm.columns = [c.replace("_bal", "") for c in heat_cols]
            hm["_mean"] = hm.mean(axis=1)
            hm = hm.sort_values("_mean", ascending=False).head(50).drop(columns="_mean")
            y_labels = [f"{x}  {SEGMENT_LABELS.get(str(x), '')}" for x in hm.index]
            fig_hm = px.imshow(
                hm.values * 100, x=list(hm.columns), y=y_labels,
                labels=dict(x="Communication", y="Segment", color="Bal% Δ"),
                color_continuous_scale="RdYlGn", aspect="auto",
                title="Balance % Δ — Top 50 segments",
            )
            fig_hm.update_layout(height=700)
            st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ── Journey timeline ─────────────────────────────────────────────────
        st.subheader("🗺️ Journey Timeline — avg balance at each touchpoint")
        st.caption(
            "📈 **Journey timeline** — tracks how selected segments perform at each "
            "communication touchpoint. Rising lines = improving engagement over the journey. "
            "Flat or falling lines suggest fatigue or declining relevance at that stage. "
            "Useful for optimising send timing and dropping ineffective touchpoints."
        )
        _jt_sort = next((f"{c}_bal" for c in ordered_comms if f"{c}_bal" in tbl.columns), None)
        jt_pool = (tbl.sort_values(_jt_sort, ascending=False).head(30).index.astype(str).tolist()
                   if _jt_sort else tbl.index.astype(str).tolist()[:30])
        jt_segs = st.multiselect(
            "Segments for timeline (top 30 by agg bal%)",
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
                            "Segment": f"{seg}  {SEGMENT_LABELS.get(seg, '')}",
                            "Avg Balance % Change": grp["balance_pct_change"].mean(),
                            "_rank": _rank(comm),
                        })
            if jt_rows:
                jt_df = pd.DataFrame(jt_rows).sort_values("_rank")
                fig_jt = px.line(
                    jt_df, x="Communication", y="Avg Balance % Change", color="Segment",
                    markers=True,
                    category_orders={"Communication": ordered_comms},
                    title="Avg Balance % Change across journey touchpoints",
                )
                fig_jt.update_yaxes(tickformat=".1%")
                fig_jt.update_layout(height=450)
                st.plotly_chart(fig_jt, use_container_width=True)
            else:
                st.info("No journey data for selected segments.")

        st.divider()

        # ── Segment co-occurrence heatmap ────────────────────────────────────
        st.subheader("🔗 Segment Co-occurrence — how often segments share the same user")
        st.caption(
            "🧩 **Co-occurrence heatmap** — cell [A, B] = fraction of segment A users "
            "who are also in segment B. Dark blue = heavy overlap. "
            "If two high-performing segments overlap heavily, targeting both wastes budget — "
            "pick the one with stronger lift. Also useful for building exclusion lists."
        )
        _cooc_sort = next((f"{c}_bal" for c in ordered_comms if f"{c}_bal" in tbl.columns), None)
        top_segs_cooc = (tbl.sort_values(_cooc_sort, ascending=False).head(30).index.astype(str).tolist()
                         if _cooc_sort else tbl.index.astype(str).tolist()[:30])
        cooc_n = st.slider("Top N segments to include", 5, min(50, len(top_segs_cooc)), min(20, len(top_segs_cooc)), key="cooc_n")
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
        y_lbl = [f"{x} {SEGMENT_LABELS.get(x, '')}" for x in cooc_segs]
        fig_cooc = px.imshow(
            cooc_norm_df.values,
            x=y_lbl, y=y_lbl,
            color_continuous_scale="Blues",
            zmin=0, zmax=1,
            labels=dict(color="Share of row-seg users"),
            title="Segment co-occurrence (row = % of row-segment users who also belong to col-segment)",
            aspect="auto",
        )
        fig_cooc.update_layout(height=600)
        st.plotly_chart(fig_cooc, use_container_width=True)
        st.caption(
            "**How to read**: cell [A, B] = fraction of segment A users who are also in segment B. "
            "High values (dark blue) mean heavy overlap — avoid targeting both segments simultaneously."
        )

        st.divider()

        # ── Violin / distribution ────────────────────────────────────────────
        st.subheader("Balance % change distribution by communication")
        st.caption(
            "🎻 **Violin chart** — shows the full spread of individual balance changes, "
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
            ].dropna(subset=["balance_pct_change"])
            if not dv.empty:
                dv = dv.copy()
                dv["_label"] = dv["nsegment"].map(lambda x: f"{x}  {SEGMENT_LABELS.get(x, '')}")
                fig_v = px.violin(
                    dv, x="Communication", y="balance_pct_change",
                    color="_label", box=True, points=False,
                    category_orders={"Communication": ordered_comms},
                    labels={"balance_pct_change": "Balance % change", "_label": "Segment"},
                    title="Balance % change distribution per communication",
                )
                fig_v.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_v, use_container_width=True)

        st.divider()

        # ── Distribution explorer ────────────────────────────────────────────
        st.subheader("Distribution explorer — KDE density")
        st.caption(
            "🔍 **Distribution explorer** — smoothed KDE density curve for any "
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
# EXPORT TAB
# ══════════════════════════════════════════════════════════
with tab_export:
    st.subheader("Export table + charts")
    if "tbl" not in dir() or tbl.empty:
        st.warning("No table data — adjust filters in the Table tab first.")
    else:
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            csv_bytes = tbl.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv_bytes, "segment_table.csv", "text/csv")
        with ec2:
            try:
                xlsx_bytes = build_excel(tbl, ordered_comms)
                st.download_button("⬇ Download Excel (.xlsx)", xlsx_bytes,
                                   "segment_table.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Excel export error: {e}")
        with ec3:
            try:
                pptx_bytes = build_pptx(tbl, ordered_comms)
                st.download_button("⬇ Download PowerPoint (.pptx)", pptx_bytes,
                                   "segment_report.pptx",
                                   "application/vnd.openxmlformats-officedocument.presentationml.presentation")
            except Exception as e:
                st.error(f"PowerPoint export error: {e}")

        st.divider()
        st.caption(
            "**Excel**: colour-coded table with conditional formatting.  \n"
            "**PowerPoint**: title slide + top-20 segments table slide.  \n"
            "**CSV**: raw numbers for further analysis."
        )


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
