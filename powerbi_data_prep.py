"""
powerbi_data_prep.py  —  Prepare Power BI-ready tables from Santander segment data.

Reads dummy_segment_data.csv + segment_descriptions.csv and outputs a set of
flat CSV tables into ./powerbi/ that can be imported directly into Power BI.

Tables produced:
  dim_segments.csv          – Segment dimension (ID, label, description, group)
  dim_communications.csv    – Communication dimension (name, sort order)
  fact_observations.csv     – Row-level exploded data (one row per user × segment × comm)
  fact_segment_stats.csv    – Pre-computed per-segment × per-communication stats
  fact_combo_pairs.csv      – Segment pair combo results (all communications)
  fact_combo_triples.csv    – Segment triple combo results (all communications)
  fact_cooccurrence.csv     – Segment co-occurrence matrix (long format)
  fact_journey.csv          – Journey timeline data (segment × communication means)

Run:  python powerbi_data_prep.py
"""

import ast
import os
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("powerbi")
COMM_ORDER = ["day1", "day5", "day7", "day31", "day61", "day90", "day120"]
Z95 = 1.96
MIN_N_DEFAULT = 30
COMBO_MIN_CUSTOMERS = 20
COMBO_CANDIDATE_POOL = 25


# ── Helpers (same logic as app.py) ────────────────────────────────────────────
def _try_parse_listlike(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None and str(v).strip() != ""]
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
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]
        if s.startswith("-") and s.endswith("-") and len(s) > 1:
            parts = [p.strip() for p in s.strip("-").split("-")]
            return [p for p in parts if p]
        return [s.strip("'").strip('"')]
    return [str(x)]


def _rank(c: str) -> int:
    try:
        return COMM_ORDER.index(c)
    except ValueError:
        return 999


def _wmean(vals, weights=None):
    mask = vals.notna()
    v = vals[mask]
    if len(v) == 0:
        return np.nan
    if weights is None:
        return float(v.mean())
    w = weights[mask]
    if w.sum() == 0:
        return np.nan
    return float(np.average(v, weights=w))


def _se(vals, weights=None):
    mask = vals.notna()
    v = vals[mask].to_numpy(dtype=float)
    n = len(v)
    if n < 2:
        return np.nan
    if weights is None:
        return float(np.std(v, ddof=1) / np.sqrt(n))
    w = weights[mask].to_numpy(dtype=float)
    if w.sum() == 0:
        return np.nan
    wm = float(np.average(v, weights=w))
    wvar = float(np.average((v - wm) ** 2, weights=w)) * n / (n - 1)
    return float(np.sqrt(wvar / n))


# ── Load & Preprocess ────────────────────────────────────────────────────────
def load_and_preprocess(
    data_path: str = "dummy_segment_data.csv",
    desc_path: str = "segment_descriptions.csv",
) -> tuple:
    """Return (df_exploded, seg_labels, seg_descriptions, seg_groups)."""

    df = pd.read_csv(data_path)
    df = df.rename(columns={"Communication": "communication", "Contact_flag": "contact_flag"})

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df["communication"] = df["communication"].astype(str).str.strip()
    df["contact_flag"]  = pd.to_numeric(df["contact_flag"], errors="coerce").fillna(0).astype(int)
    if "control_flag" in df.columns:
        df["control_flag"] = pd.to_numeric(df["control_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["control_flag"] = (df["contact_flag"] == 0).astype(int)

    for col in ["start_balance", "end_balance", "start_accounts", "end_accounts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Explode segments
    df["nsegments"] = df["nsegments"].apply(_try_parse_listlike)
    df["nsegments"] = df["nsegments"].apply(lambda xs: xs if xs else ["__NO_SEGMENT__"])
    df = df.explode("nsegments").rename(columns={"nsegments": "nsegment"})
    df["nsegment"] = df["nsegment"].astype(str)

    # Computed columns
    eps = 1e-6
    df["balance_abs_change"]  = df["end_balance"] - df["start_balance"]
    denom_b = df["start_balance"].abs().replace(0, np.nan)
    df["balance_pct_change"]  = df["balance_abs_change"] / denom_b
    df.loc[df["start_balance"].abs() < eps, "balance_pct_change"] = np.nan

    df["accounts_abs_change"] = df["end_accounts"] - df["start_accounts"]
    denom_a = df["start_accounts"].replace(0, np.nan)
    df["accounts_pct_change"] = df["accounts_abs_change"] / denom_a
    df.loc[df["start_accounts"].fillna(0) == 0, "accounts_pct_change"] = np.nan

    # Clip extreme balance % at 5th/95th percentile
    lo = df["balance_pct_change"].quantile(0.05)
    hi = df["balance_pct_change"].quantile(0.95)
    df["balance_pct_change"] = df["balance_pct_change"].clip(lo, hi)

    # Segment metadata
    seg_labels = {}
    seg_descriptions = {}
    seg_groups = {}
    try:
        sd = pd.read_csv(desc_path, dtype=str)
        if {"nsegment", "label", "description"}.issubset(sd.columns):
            seg_labels = dict(zip(sd["nsegment"], sd["label"].fillna("")))
            seg_descriptions = dict(zip(sd["nsegment"], sd["description"].fillna("")))
            for _, row in sd.iterrows():
                seg_groups.setdefault(row["label"], []).append(row["nsegment"])
    except FileNotFoundError:
        pass

    return df, seg_labels, seg_descriptions, seg_groups


# ── Build dimension tables ───────────────────────────────────────────────────
def build_dim_segments(seg_labels, seg_descriptions) -> pd.DataFrame:
    segs = sorted(set(list(seg_labels.keys()) + list(seg_descriptions.keys())))
    return pd.DataFrame({
        "nsegment":    segs,
        "label":       [seg_labels.get(s, "") for s in segs],
        "description": [seg_descriptions.get(s, "") for s in segs],
        "group":       [seg_labels.get(s, "") for s in segs],
    })


def build_dim_communications(df: pd.DataFrame) -> pd.DataFrame:
    present = sorted(df["communication"].unique(), key=_rank)
    return pd.DataFrame({
        "communication": present,
        "sort_order": [_rank(c) for c in present],
    })


# ── Build fact_observations ──────────────────────────────────────────────────
def build_fact_observations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "alpha_key", "nsegment", "communication",
        "contact_flag", "control_flag",
        "start_date", "end_date",
        "start_balance", "end_balance",
        "start_accounts", "end_accounts",
        "balance_abs_change", "balance_pct_change",
        "accounts_abs_change", "accounts_pct_change",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


# ── Build fact_segment_stats ─────────────────────────────────────────────────
def build_fact_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment × per-communication statistics: means, lifts, CIs, N."""
    comms = sorted(df["communication"].unique(), key=_rank)
    rows = []

    for comm in comms:
        treated = df[(df["contact_flag"] == 1) & (df["communication"] == comm)]
        control = df[(df["control_flag"] == 1) & (df["communication"] == comm)]

        for seg, grp in treated.groupby("nsegment"):
            n_t = grp["alpha_key"].nunique()
            mb  = _wmean(grp["balance_pct_change"])
            ma  = _wmean(grp["accounts_pct_change"])
            se_b = _se(grp["balance_pct_change"])
            se_a = _se(grp["accounts_pct_change"])
            ci_b = Z95 * se_b if not np.isnan(se_b) else np.nan
            ci_a = Z95 * se_a if not np.isnan(se_a) else np.nan

            ctrl = control[control["nsegment"] == seg]
            n_c  = ctrl["alpha_key"].nunique() if not ctrl.empty else 0
            cb   = _wmean(ctrl["balance_pct_change"]) if not ctrl.empty else np.nan
            ca   = _wmean(ctrl["accounts_pct_change"]) if not ctrl.empty else np.nan
            se_cb = _se(ctrl["balance_pct_change"]) if not ctrl.empty else np.nan
            se_ca = _se(ctrl["accounts_pct_change"]) if not ctrl.empty else np.nan

            lift_b = (mb - cb) if not (np.isnan(mb) or np.isnan(cb)) else np.nan
            lift_a = (ma - ca) if not (np.isnan(ma) or np.isnan(ca)) else np.nan
            lift_ci_b = float(Z95 * np.sqrt(se_b**2 + se_cb**2)) if not (np.isnan(se_b) or np.isnan(se_cb)) else np.nan
            lift_ci_a = float(Z95 * np.sqrt(se_a**2 + se_ca**2)) if not (np.isnan(se_a) or np.isnan(se_ca)) else np.nan

            # Average starting balance/accounts for projection
            per_user = grp.drop_duplicates("alpha_key")
            avg_start_bal  = per_user["start_balance"].mean()
            avg_start_acct = per_user["start_accounts"].mean()

            rows.append({
                "nsegment": seg,
                "communication": comm,
                "comm_sort_order": _rank(comm),
                "n_treated": n_t,
                "n_control": n_c,
                "treated_mean_bal": mb,
                "treated_ci_bal": ci_b,
                "treated_mean_acct": ma,
                "treated_ci_acct": ci_a,
                "control_mean_bal": cb,
                "control_mean_acct": ca,
                "lift_bal": lift_b,
                "lift_ci_bal": lift_ci_b,
                "lift_acct": lift_a,
                "lift_ci_acct": lift_ci_a,
                "lift_bal_significant": (
                    1 if (pd.notna(lift_b) and pd.notna(lift_ci_b)
                          and ((lift_b - lift_ci_b) > 0 or (lift_b + lift_ci_b) < 0))
                    else 0
                ),
                "lift_acct_significant": (
                    1 if (pd.notna(lift_a) and pd.notna(lift_ci_a)
                          and ((lift_a - lift_ci_a) > 0 or (lift_a + lift_ci_a) < 0))
                    else 0
                ),
                "avg_start_balance": avg_start_bal,
                "avg_start_accounts": avg_start_acct,
                "projected_bal_increase": (n_t * avg_start_bal * lift_b
                                           if pd.notna(lift_b) and pd.notna(avg_start_bal) else np.nan),
                "projected_acct_increase": (n_t * avg_start_acct * lift_a
                                            if pd.notna(lift_a) and pd.notna(avg_start_acct) else np.nan),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Build fact_cooccurrence (long format) ────────────────────────────────────
def build_fact_cooccurrence(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    """Segment co-occurrence as long-format table for Power BI matrix visual."""
    seg_counts = df.drop_duplicates(["alpha_key", "nsegment"]).groupby("nsegment")["alpha_key"].nunique()
    top_segs = seg_counts.sort_values(ascending=False).head(top_n).index.tolist()

    indicator = (
        df[df["nsegment"].isin(top_segs)]
        .drop_duplicates(subset=["alpha_key", "nsegment"])
        .assign(_v=1)
        .pivot_table(index="alpha_key", columns="nsegment", values="_v", fill_value=0)
    )
    indicator = indicator.reindex(columns=top_segs, fill_value=0)
    vals = indicator.values
    cooc = vals.T @ vals  # (n_segs × n_segs) counts

    diag = np.diag(cooc).copy()
    diag[diag == 0] = 1
    cooc_norm = cooc / diag[:, None]

    rows = []
    for i, seg_a in enumerate(top_segs):
        for j, seg_b in enumerate(top_segs):
            rows.append({
                "segment_a": seg_a,
                "segment_b": seg_b,
                "shared_customers": int(cooc[i, j]),
                "overlap_pct": float(cooc_norm[i, j]),
            })
    return pd.DataFrame(rows)


# ── Build fact_journey ───────────────────────────────────────────────────────
def build_fact_journey(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Per-segment × per-communication means for journey timeline visual."""
    comms = sorted(df["communication"].unique(), key=_rank)
    treated = df[df["contact_flag"] == 1]

    # Pick top segments by overall treated mean balance change
    seg_means = treated.groupby("nsegment")["balance_pct_change"].mean()
    top_segs = seg_means.sort_values(ascending=False).head(top_n).index.tolist()

    rows = []
    for comm in comms:
        sub = treated[treated["communication"] == comm]
        for seg in top_segs:
            grp = sub[sub["nsegment"] == seg]
            if grp.empty:
                continue
            rows.append({
                "nsegment": seg,
                "communication": comm,
                "comm_sort_order": _rank(comm),
                "mean_bal_pct": grp["balance_pct_change"].mean(),
                "mean_acct_pct": grp["accounts_pct_change"].mean(),
                "n_customers": grp["alpha_key"].nunique(),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Build segment combo tables ───────────────────────────────────────────────
def _build_combos(
    df: pd.DataFrame,
    comm: str,
    metric_col: str,
    lift_col_name: str,
    stats_df: pd.DataFrame,
    combo_size: int,
    min_customers: int,
    candidate_pool: int,
) -> List[dict]:
    """Find segment combos for one communication."""
    # Get individual lifts from stats_df
    comm_stats = stats_df[stats_df["communication"] == comm]
    ind_lifts = comm_stats.set_index("nsegment")[lift_col_name].dropna()
    if ind_lifts.empty:
        return []

    candidate_segs = ind_lifts.sort_values(ascending=False).head(candidate_pool).index.tolist()

    treated = df[(df["contact_flag"] == 1) & (df["communication"] == comm)]
    control = df[(df["control_flag"] == 1) & (df["communication"] == comm)]

    def _build_inputs(data, segs):
        pairs = data.drop_duplicates(subset=["alpha_key", "nsegment"])[["alpha_key", "nsegment"]]
        pairs = pairs[pairs["nsegment"].isin(segs)]
        if pairs.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        ind = pairs.assign(_v=1).pivot_table(
            index="alpha_key", columns="nsegment", values="_v", fill_value=0
        )
        ind = ind.reindex(columns=segs, fill_value=0)
        outcomes = data.drop_duplicates("alpha_key").set_index("alpha_key")[metric_col]
        return ind, outcomes

    t_ind, t_out = _build_inputs(treated, candidate_segs)
    c_ind, c_out = _build_inputs(control, candidate_segs)

    if t_ind.empty:
        return []

    t_arr = t_ind.values
    c_arr = c_ind.values if not c_ind.empty else np.empty((0, len(candidate_segs)))

    combo_rows = []
    for combo_idx in combinations(range(len(candidate_segs)), combo_size):
        t_mask = t_arr[:, list(combo_idx)].all(axis=1)
        n_t = int(t_mask.sum())
        if n_t < min_customers:
            continue
        t_users = t_ind.index[t_mask]
        t_vals = t_out.reindex(t_users).dropna()
        if t_vals.empty:
            continue
        t_mean = float(t_vals.mean())

        if c_arr.shape[0] > 0:
            c_mask = c_arr[:, list(combo_idx)].all(axis=1)
            c_users = c_ind.index[c_mask]
            c_vals = c_out.reindex(c_users).dropna()
            c_mean = float(c_vals.mean()) if len(c_vals) >= 5 else np.nan
        else:
            c_mean = np.nan

        combo_lift = (t_mean - c_mean) if not np.isnan(c_mean) else np.nan
        combo_segs = [candidate_segs[i] for i in combo_idx]
        best_ind = max((ind_lifts.get(s, np.nan) for s in combo_segs), default=np.nan)
        synergy = (combo_lift - best_ind) if (pd.notna(combo_lift) and pd.notna(best_ind)) else np.nan

        combo_rows.append({
            "segments": " + ".join(combo_segs),
            "communication": comm,
            "combo_lift_bal": combo_lift if metric_col == "balance_pct_change" else np.nan,
            "combo_lift_acct": combo_lift if metric_col == "accounts_pct_change" else np.nan,
            "best_individual_lift": best_ind,
            "synergy": synergy,
            "n_customers": n_t,
        })
    return combo_rows


def build_fact_combos(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    combo_size: int = 2,
) -> pd.DataFrame:
    """Build combo results for all communications."""
    comms = sorted(df["communication"].unique(), key=_rank)
    all_rows = []
    for comm in comms:
        # Balance lift combos
        bal_rows = _build_combos(
            df, comm, "balance_pct_change", "lift_bal", stats_df,
            combo_size, COMBO_MIN_CUSTOMERS, COMBO_CANDIDATE_POOL,
        )
        all_rows.extend(bal_rows)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    df, seg_labels, seg_descriptions, seg_groups = load_and_preprocess()
    print(f"  Rows: {len(df):,}  Users: {df['alpha_key'].nunique():,}  Segments: {df['nsegment'].nunique()}")

    # 1. Dimension tables
    print("Building dim_segments...")
    dim_seg = build_dim_segments(seg_labels, seg_descriptions)
    dim_seg.to_csv(OUT_DIR / "dim_segments.csv", index=False)
    print(f"  → {len(dim_seg)} segments")

    print("Building dim_communications...")
    dim_comm = build_dim_communications(df)
    dim_comm.to_csv(OUT_DIR / "dim_communications.csv", index=False)
    print(f"  → {len(dim_comm)} communications")

    # 2. Fact: observations
    print("Building fact_observations...")
    fact_obs = build_fact_observations(df)
    fact_obs.to_csv(OUT_DIR / "fact_observations.csv", index=False)
    print(f"  → {len(fact_obs):,} rows")

    # 3. Fact: segment stats
    print("Building fact_segment_stats...")
    stats = build_fact_segment_stats(df)
    stats.to_csv(OUT_DIR / "fact_segment_stats.csv", index=False)
    print(f"  → {len(stats):,} segment × communication cells")

    # 4. Fact: co-occurrence
    print("Building fact_cooccurrence...")
    cooc = build_fact_cooccurrence(df)
    cooc.to_csv(OUT_DIR / "fact_cooccurrence.csv", index=False)
    print(f"  → {len(cooc):,} pairs")

    # 5. Fact: journey
    print("Building fact_journey...")
    journey = build_fact_journey(df)
    journey.to_csv(OUT_DIR / "fact_journey.csv", index=False)
    print(f"  → {len(journey):,} rows")

    # 6. Fact: combos (pairs)
    print("Building fact_combo_pairs...")
    combos2 = build_fact_combos(df, stats, combo_size=2)
    combos2.to_csv(OUT_DIR / "fact_combo_pairs.csv", index=False)
    print(f"  → {len(combos2):,} pairs")

    # 7. Fact: combos (triples)
    print("Building fact_combo_triples...")
    combos3 = build_fact_combos(df, stats, combo_size=3)
    combos3.to_csv(OUT_DIR / "fact_combo_triples.csv", index=False)
    print(f"  → {len(combos3):,} triples")

    print(f"\nAll files written to ./{OUT_DIR}/")
    print("Import these CSVs into Power BI via Get Data → Text/CSV.")


if __name__ == "__main__":
    main()
