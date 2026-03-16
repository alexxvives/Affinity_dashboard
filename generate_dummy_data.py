from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
import pandas as pd
import random


def generate_dummy_dataset(n_rows=25000, min_users=200, max_users=500, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    n_users = int(np.random.randint(min_users, max_users + 1))

    _comm_list = ['day1', 'day5', 'day7', 'day31', 'day61', 'day90', 'day120']

    # Non-uniform send volumes: day1 and day90 are large waves, day5 is light
    comm_weights = {
        'day1': 0.24, 'day5': 0.06, 'day7': 0.11,
        'day31': 0.15, 'day61': 0.13, 'day90': 0.20, 'day120': 0.11,
    }
    # Per-communication contact rate
    comm_contact_rate = {
        'day1': 0.80, 'day5': 0.60, 'day7': 0.65,
        'day31': 0.70, 'day61': 0.72, 'day90': 0.75, 'day120': 0.68,
    }
    # Per-communication treatment balance effect (mean, std)
    comm_fx = {
        'day1': (320.0, 300.0), 'day5': (90.0, 250.0), 'day7': (150.0, 280.0),
        'day31': (230.0, 320.0), 'day61': (260.0, 340.0), 'day90': (290.0, 360.0),
        'day120': (200.0, 300.0),
    }

    segment_pool = [f'{i:07d}' for i in range(1000000, 1000150)]
    base_date = datetime(2024, 1, 1)

    users = [f'user_{i}' for i in range(n_users)]
    user_segments = {u: random.sample(segment_pool, int(np.random.randint(1, 5))) for u in users}

    # User-communication affinity cohorts for realistic N variation across comms
    user_cohort = {u: int(np.random.randint(0, 4)) for u in users}
    cohort_w = [
        {'day1': 0.30, 'day5': 0.04, 'day7': 0.08, 'day31': 0.10, 'day61': 0.10, 'day90': 0.28, 'day120': 0.10},
        {'day1': 0.10, 'day5': 0.10, 'day7': 0.15, 'day31': 0.25, 'day61': 0.25, 'day90': 0.10, 'day120': 0.05},
        {'day1': 0.14, 'day5': 0.14, 'day7': 0.14, 'day31': 0.15, 'day61': 0.15, 'day90': 0.14, 'day120': 0.14},
        {'day1': 0.05, 'day5': 0.05, 'day7': 0.05, 'day31': 0.10, 'day61': 0.15, 'day90': 0.30, 'day120': 0.30},
    ]

    user_bal = {u: max(0.0, float(np.random.normal(3000.0, 1500.0))) for u in users}
    user_acct = {u: int(np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])) for u in users}

    gw = np.array([comm_weights[c] for c in _comm_list])
    gw = gw / gw.sum()

    rows = []
    for _ in range(n_rows):
        u = random.choice(users)
        coh = user_cohort[u]
        cw = np.array([cohort_w[coh][c] for c in _comm_list])
        cw = cw / cw.sum()
        bl = 0.6 * gw + 0.4 * cw
        bl = bl / bl.sum()
        comm = np.random.choice(_comm_list, p=bl)
        cr = comm_contact_rate[comm]
        cf = int(np.random.choice([0, 1], p=[1 - cr, cr]))
        sd = base_date + timedelta(days=int(np.random.randint(0, 180)))
        ed = sd + timedelta(days=7)
        sb = max(0.0, float(np.random.normal(user_bal[u], 500.0)))
        sa = int(user_acct[u])
        fm, fs = comm_fx[comm]
        if cf == 1:
            bc = float(np.random.normal(fm, fs))
            ac = int(np.random.choice([0, 1], p=[0.75, 0.25]))
        else:
            bc = float(np.random.normal(50.0, 200.0))
            ac = int(np.random.choice([0, 1], p=[0.95, 0.05]))
        eb = max(0.0, sb + bc)
        ea = max(0, sa + ac)
        rows.append({
            'Communication': comm,
            'alpha_key': u,
            'start_date': sd,
            'end_date': ed,
            'start_balance': round(sb, 2),
            'end_balance': round(eb, 2),
            'start_accounts': int(sa),
            'end_accounts': int(ea),
            'nsegments': user_segments[u],
            'Contact_flag': cf,
            'control_flag': 1 - cf,
        })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df_dummy = generate_dummy_dataset(n_rows=25000, min_users=200, max_users=500, seed=42)
    df_dummy.to_csv('dummy_segment_data.csv', index=False)
    print('Dummy dataset created: dummy_segment_data.csv')
    print('Rows:', len(df_dummy), '  Users:', df_dummy['alpha_key'].nunique())
    print(df_dummy.groupby('Communication')['alpha_key'].count().sort_values(ascending=False))
