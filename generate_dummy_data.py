from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
import pandas as pd
import random


# Segment labels / descriptions used by the Streamlit app when segment_descriptions.csv is absent
SEGMENT_LABELS: Dict[str, str] = {}
SEGMENT_DESCRIPTIONS: Dict[str, str] = {}


def generate_dummy_dataset(n_users=5000, n_rows=100000, seed=42):
    """Generate a realistic dummy dataset with enough segment overlap for combo analysis.

    Key design decisions vs the old generator:
    - 5,000 users (was 200-500) so segment-pair intersections are substantial
    - 40 segments (was 150) to ensure overlap
    - Users each belong to 2-6 segments (drawn from a biased pool so frequent segments overlap heavily)
    - Some segment pairs have hard-coded positive synergy effects so the combo explorer finds them
    - 100,000 rows (was 25,000) to match realistic production scale
    """
    np.random.seed(seed)
    random.seed(seed)

    _comm_list = ['day1', 'day5', 'day7', 'day31', 'day61DD', 'day61NDD', 'day90', 'day120']

    comm_weights = {
        'day1': 0.24, 'day5': 0.06, 'day7': 0.11,
        'day31': 0.15, 'day61DD': 0.07, 'day61NDD': 0.06, 'day90': 0.20, 'day120': 0.11,
    }
    comm_contact_rate = {
        'day1': 0.80, 'day5': 0.60, 'day7': 0.65,
        'day31': 0.70, 'day61DD': 0.72, 'day61NDD': 0.68, 'day90': 0.75, 'day120': 0.68,
    }
    # (mean €, std €) balance increase for treated group per communication
    comm_fx = {
        'day1':    (320.0, 300.0), 'day5':  ( 90.0, 250.0), 'day7':  (150.0, 280.0),
        'day31':   (230.0, 320.0), 'day61DD': (260.0, 340.0), 'day61NDD': (240.0, 330.0),
        'day90':   (290.0, 360.0), 'day120': (200.0, 300.0),
    }

    # ── Segment pool ────────────────────────────────────────────────────────
    # 40 segments; first 10 are "high frequency" and appear in many users
    segment_pool = [f'{i:07d}' for i in range(1000000, 1000040)]
    # Biased draw: first 10 segments are 3× more likely than the rest
    seg_probs = np.array([3.0 if i < 10 else 1.0 for i in range(len(segment_pool))])
    seg_probs = seg_probs / seg_probs.sum()

    # ── Per-segment balance effect multiplier (treatment) ──────────────────
    # Some segments respond much better to treatment (lift multiplier on comm_fx mean)
    seg_fx = {s: float(np.random.uniform(0.6, 1.8)) for s in segment_pool}
    # Intentionally bake in strong per-segment effects for first 10 segments
    strong_segs = segment_pool[:10]
    for s in strong_segs:
        seg_fx[s] = float(np.random.uniform(1.4, 2.2))

    # ── Synergy pairs/triples: combo of these two segments → extra boost ───
    # These are planted so the Segment Combo Explorer has something real to find
    synergy_pairs = {
        (segment_pool[0], segment_pool[1]): 400.0,   # very strong pair
        (segment_pool[0], segment_pool[2]): 250.0,
        (segment_pool[1], segment_pool[3]): 180.0,
        (segment_pool[2], segment_pool[4]): 300.0,
        (segment_pool[5], segment_pool[6]): 220.0,
        (segment_pool[3], segment_pool[7]): 150.0,
        # A triple with synergy
        (segment_pool[0], segment_pool[1], segment_pool[2]): 600.0,
    }

    base_date = datetime(2024, 1, 1)
    users = [f'user_{i:05d}' for i in range(n_users)]

    # Each user draws 2-6 segments from the biased pool (without replacement)
    n_segs_per_user = np.random.choice([2, 3, 4, 5, 6], size=n_users, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    user_segments: Dict[str, List[str]] = {}
    for u, k in zip(users, n_segs_per_user):
        chosen_indices = np.random.choice(len(segment_pool), size=int(k), replace=False, p=seg_probs)
        user_segments[u] = [segment_pool[i] for i in chosen_indices]

    user_cohort = {u: int(np.random.randint(0, 4)) for u in users}
    cohort_w = [
        {'day1': 0.30, 'day5': 0.04, 'day7': 0.08, 'day31': 0.10, 'day61DD': 0.06, 'day61NDD': 0.04, 'day90': 0.28, 'day120': 0.10},
        {'day1': 0.10, 'day5': 0.10, 'day7': 0.15, 'day31': 0.25, 'day61DD': 0.14, 'day61NDD': 0.11, 'day90': 0.10, 'day120': 0.05},
        {'day1': 0.14, 'day5': 0.14, 'day7': 0.14, 'day31': 0.14, 'day61DD': 0.08, 'day61NDD': 0.07, 'day90': 0.14, 'day120': 0.15},
        {'day1': 0.05, 'day5': 0.05, 'day7': 0.05, 'day31': 0.10, 'day61DD': 0.08, 'day61NDD': 0.07, 'day90': 0.30, 'day120': 0.30},
    ]

    user_bal  = {u: max(200.0, float(np.random.normal(3000.0, 1500.0))) for u in users}
    user_acct = {u: int(np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])) for u in users}

    gw = np.array([comm_weights[c] for c in _comm_list])
    gw = gw / gw.sum()

    def _seg_synergy_bonus(segs: List[str], treated: bool) -> float:
        if not treated:
            return 0.0
        seg_set = frozenset(segs)
        for pair, bonus in synergy_pairs.items():
            if frozenset(pair).issubset(seg_set):
                return bonus
        return 0.0

    rows = []
    users_arr = np.array(users)
    for _ in range(n_rows):
        u = users_arr[np.random.randint(n_users)]
        coh = user_cohort[u]
        cw = np.array([cohort_w[coh][c] for c in _comm_list])
        cw = cw / cw.sum()
        bl = 0.6 * gw + 0.4 * cw
        bl = bl / bl.sum()
        comm = _comm_list[np.searchsorted(np.cumsum(bl), np.random.random())]

        cr = comm_contact_rate[comm]
        cf = 1 if np.random.random() < cr else 0
        sd = base_date + timedelta(days=int(np.random.randint(0, 365)))
        ed = sd + timedelta(days=7)
        sb = max(0.0, float(np.random.normal(user_bal[u], 500.0)))
        sa = int(user_acct[u])

        fm, fs = comm_fx[comm]
        u_segs = user_segments[u]
        if cf == 1:
            # Segment-specific multiplier + synergy bonus
            best_seg_mult = max(seg_fx.get(s, 1.0) for s in u_segs)
            synergy = _seg_synergy_bonus(u_segs, True)
            bc = float(np.random.normal(fm * best_seg_mult + synergy, fs))
            ac = int(np.random.choice([0, 1], p=[0.75, 0.25]))
        else:
            bc = float(np.random.normal(50.0, 200.0))
            ac = int(np.random.choice([0, 1], p=[0.95, 0.05]))

        eb = max(0.0, sb + bc)
        ea = max(0, sa + ac)
        rows.append({
            'Communication':  comm,
            'alpha_key':      u,
            'start_date':     sd.strftime('%Y-%m-%d'),
            'end_date':       ed.strftime('%Y-%m-%d'),
            'start_balance':  round(sb, 2),
            'end_balance':    round(eb, 2),
            'start_accounts': int(sa),
            'end_accounts':   int(ea),
            'nsegments':      user_segments[u],
            'Contact_flag':   cf,
            'control_flag':   1 - cf,
        })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


PRODUCT_COLS = [
    "Business Line Of Credit", "Cdira", "Checking", "Commercial Loan",
    "Credit Card", "Escrow", "Standalone Savings", "Home Equity",
    "Investments", "Loan", "Loan - Personal", "Loc - Personal",
    "Money Market", "Mortgage", "Odloc", "Other", "Savings",
]

_US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
]
# Rough population-weighted state probabilities (larger states more likely)
_STATE_WEIGHTS = [
    1,1,2,1,10,2,1,1,7,4,1,1,5,2,1,
    1,1,1,1,2,2,3,2,1,2,1,1,1,1,3,
    1,7,3,1,4,1,1,4,1,2,1,2,9,1,1,
    3,3,1,2,1,
]


def generate_audience_profile_data(n_users: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate one demographic row per user to pair with the main transactional dataset."""
    rng = np.random.default_rng(seed)

    users = [f"user_{i:05d}" for i in range(n_users)]
    today = datetime(2026, 3, 23)

    # ── Core dates ────────────────────────────────────────────────────────────
    # Ages roughly 22-72 years old
    dob_offsets = rng.integers(int(365.25 * 22), int(365.25 * 72), size=n_users)
    dates_of_birth = [today - timedelta(days=int(d)) for d in dob_offsets]

    # Tenure 0.5 – 15 years, skewed toward shorter tenures
    tenure_days = rng.integers(180, int(365.25 * 15), size=n_users)
    date_first_relation = [today - timedelta(days=int(d)) for d in tenure_days]

    # ── Behavioral & demographic flags ───────────────────────────────────────
    # 68% active on mobile/ROB in last 90 days; slightly correlated with younger age
    age_years = dob_offsets / 365.25
    active_base_prob = np.clip(0.80 - 0.004 * (age_years - 35), 0.30, 0.92)
    flag_active = (rng.random(n_users) < active_base_prob).astype(int)

    _gender_raw = rng.choice(["Male", "Female"], size=n_users, p=[0.49, 0.51])
    # ~6 % missing gender — realistic for real data
    _gender_missing_mask = rng.random(n_users) < 0.06
    genders = np.where(_gender_missing_mask, None, _gender_raw)

    state_probs = np.array(_STATE_WEIGHTS, dtype=float)
    state_probs /= state_probs.sum()
    states = rng.choice(_US_STATES, size=n_users, p=state_probs)

    # ── Financial columns (log-normal to mimic real wealth distributions) ─────
    deposit_bal = np.exp(rng.normal(np.log(45_000), 1.2, size=n_users)).clip(100, 2_000_000)
    # IXI (net worth) is always ≥ deposit balance, generally much larger
    ixi_multiplier = np.exp(rng.normal(1.5, 0.8, size=n_users)).clip(1.0, 50.0)
    total_ixi = deposit_bal * ixi_multiplier

    # ── Product flags ─────────────────────────────────────────────────────────
    # Each product has a realistic adoption rate; products are mildly correlated
    # through a latent "wealth" factor
    wealth_factor = np.log(deposit_bal) / np.log(45_000)  # centred near 1.0
    product_probs = {
        "Checking":                np.clip(0.82 * wealth_factor ** 0.1, 0.50, 0.95),
        "Savings":                 np.clip(0.60 * wealth_factor ** 0.2, 0.30, 0.85),
        "Credit Card":             np.clip(0.55 * wealth_factor ** 0.3, 0.20, 0.80),
        "Mortgage":                np.clip(0.28 * wealth_factor ** 0.5, 0.05, 0.60),
        "Home Equity":             np.clip(0.18 * wealth_factor ** 0.6, 0.03, 0.45),
        "Investments":             np.clip(0.22 * wealth_factor ** 0.8, 0.04, 0.55),
        "Money Market":            np.clip(0.15 * wealth_factor ** 0.7, 0.03, 0.40),
        "Loan":                    np.clip(0.20 * wealth_factor ** 0.3, 0.05, 0.50),
        "Loan - Personal":         np.clip(0.16 * wealth_factor ** 0.2, 0.04, 0.45),
        "Loc - Personal":          np.clip(0.12 * wealth_factor ** 0.3, 0.03, 0.35),
        "Standalone Savings":      np.clip(0.10 * wealth_factor ** 0.4, 0.02, 0.30),
        "Business Line Of Credit": np.clip(0.08 * wealth_factor ** 0.9, 0.01, 0.25),
        "Commercial Loan":         np.clip(0.06 * wealth_factor ** 0.9, 0.01, 0.20),
        "Cdira":                   np.clip(0.12 * wealth_factor ** 0.5, 0.02, 0.30),
        "Escrow":                  np.clip(0.09 * wealth_factor ** 0.6, 0.01, 0.25),
        "Odloc":                   np.clip(0.07 * wealth_factor ** 0.4, 0.01, 0.20),
        "Other":                   np.clip(0.14 * wealth_factor ** 0.2, 0.03, 0.35),
    }
    flags = {
        col: (rng.random(n_users) < product_probs[col]).astype(int)
        for col in PRODUCT_COLS
    }

    df = pd.DataFrame({
        "alpha_key":                    users,
        "date_first_relation":          [d.strftime("%Y-%m-%d") for d in date_first_relation],
        "date_of_birth":                [d.strftime("%Y-%m-%d") for d in dates_of_birth],
        "flag_last90_active_mob_rob":   flag_active,
        "gender":                       genders,
        "state":                        states,
        "amount_deposit_spot_balance":  deposit_bal.round(2),
        "total_deposits_ixi":           total_ixi.round(2),
        **flags,
    })

    # ── Feature engineering ───────────────────────────────────────────────────
    df["age"] = ((today - pd.to_datetime(df["date_of_birth"])).dt.days / 365.25).round(1)
    df["tenure_years"] = ((today - pd.to_datetime(df["date_first_relation"])).dt.days / 365.25).round(2)
    df["sow"] = (df["amount_deposit_spot_balance"] / df["total_deposits_ixi"]).clip(0, 1).round(4)
    df["n_products"] = df[PRODUCT_COLS].sum(axis=1).astype(int)

    return df


if __name__ == '__main__':
    df_dummy = generate_dummy_dataset(n_users=5000, n_rows=100000, seed=42)
    df_dummy.to_csv('data.csv', index=False)
    print('Dummy dataset written: data.csv')
    print(f"Rows: {len(df_dummy):,}   Users: {df_dummy['alpha_key'].nunique():,}")
    print(df_dummy.groupby('Communication')['alpha_key'].count().sort_values(ascending=False))

    aud = generate_audience_profile_data(n_users=5000, seed=42)
    aud.to_csv('audience_profile.csv', index=False)
    print(f"\nAudience profile written: audience_profile.csv")
    print(f"Rows: {len(aud):,}   Columns: {list(aud.columns)}")
    print(aud[["age", "tenure_years", "sow", "n_products"]].describe().round(2).to_string())
