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


def generate_cc_bt_campaign_data(n_users: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate one row per user for the Credit Card Balance Transfer campaign.

    Campaign goal: convince customers to transfer existing debt to our credit card.
    - control_flag=1  → control group (did not receive the BT offer)
    - control_flag=0  → treated group (received the BT offer)
    - BT_Transaction_Amount → USD amount transferred (0 when no transfer occurred)
    - nsegments       → list of segment IDs this customer belongs to (same pool as SELECTCHK)
    """
    rng = np.random.default_rng(seed)

    segment_pool = [f'{i:07d}' for i in range(1000000, 1000040)]
    seg_probs = np.array([3.0 if i < 10 else 1.0 for i in range(len(segment_pool))], dtype=float)
    seg_probs /= seg_probs.sum()

    users = [f"user_{i:05d}" for i in range(n_users)]

    # ~35 % control, ~65 % treated
    control_flag = (rng.random(n_users) < 0.35).astype(int)

    # Base BT conversion rates: 22 % treated, 7 % control (organic)
    bt_prob = np.where(control_flag == 0, 0.22, 0.07)
    BT_flag = (rng.random(n_users) < bt_prob).astype(int)

    # BT amount: log-normal centred around $3 500 for treated, $2 000 for control
    raw_amount = np.exp(
        rng.normal(
            np.where(control_flag == 0, np.log(3_500), np.log(2_000)),
            0.9,
            size=n_users,
        )
    ).clip(500, 25_000)
    BT_amount = (raw_amount * BT_flag).round(2)

    # Each user draws 2-6 segments from the same biased pool as SELECTCHK
    n_segs = rng.choice([2, 3, 4, 5, 6], size=n_users, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    nsegments = []
    for k in n_segs:
        chosen = rng.choice(len(segment_pool), size=int(k), replace=False, p=seg_probs)
        nsegments.append([segment_pool[i] for i in chosen])

    return pd.DataFrame({
        "alpha_key":             users,
        "control_flag":          control_flag,
        "BT_Transaction_Amount": BT_amount,
        "nsegments":             nsegments,
        "BT_Transaction_Date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 365, size=n_users), unit="D"
        ),
    })


if __name__ == '__main__':
    df_dummy = generate_dummy_dataset(n_users=5000, n_rows=100000, seed=42)
    df_dummy.to_csv('SELECTCHK_campaign.csv', index=False)
    print('SELECTCHK campaign written: SELECTCHK_campaign.csv')
    print(f"Rows: {len(df_dummy):,}   Users: {df_dummy['alpha_key'].nunique():,}")
    print(df_dummy.groupby('Communication')['alpha_key'].count().sort_values(ascending=False))

    cc_bt = generate_cc_bt_campaign_data(n_users=5000, seed=42)
    cc_bt.to_csv('CC_BT_campaign.csv', index=False)
    print(f"\nCC_BT campaign written: CC_BT_campaign.csv")
    treated = cc_bt[cc_bt['control_flag'] == 0]
    control = cc_bt[cc_bt['control_flag'] == 1]
    bt_rate_t = (treated['BT_Transaction_Amount'] > 0).mean()
    bt_rate_c = (control['BT_Transaction_Amount'] > 0).mean()
    print(f"Rows: {len(cc_bt):,}   BT rate treated: {bt_rate_t:.1%}   BT rate control: {bt_rate_c:.1%}")
