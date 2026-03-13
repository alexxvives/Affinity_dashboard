"""
Creates a realistic dummy dataset for the Segment Effect Explorer app.

Output:
    dummy_segment_data.csv
    segment_descriptions.csv

Usage:
    python generate_dummy_data.py
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Banking segment definitions ───────────────────────────────────────────────
# Each archetype has 10 sub-variants so all 150 IDs carry a unique description.
_SEGMENT_ARCHETYPES = [
    ("High-net-worth", range(1000000, 1000010), [
        "Balance >EUR50k - premium card holder - low digital engagement",
        "Balance >EUR50k - active investor - securities account linked",
        "Balance >EUR50k - mortgage holder - estate planning interest",
        "Balance >EUR50k - recent large deposit - liquidity event",
        "Balance >EUR50k - multi-currency - regular international transfers",
        "Balance >EUR50k - business owner - dual personal and SME accounts",
        "Balance >EUR50k - frequent FX transactions - high travel spend",
        "Balance >EUR50k - long tenure 10yr+ - loyal private client",
        "Balance >EUR50k - new to segment - recent significant balance uplift",
        "Balance >EUR50k - pension drawdown phase - income reinvestment",
    ]),
    ("Mass affluent", range(1000010, 1000020), [
        "Balance EUR10k-50k - growing savings trajectory - monthly surplus",
        "Balance EUR10k-50k - active fund purchases - medium risk appetite",
        "Balance EUR10k-50k - property purchase planned - mortgage pending",
        "Balance EUR10k-50k - stable mid-career professional - regular income",
        "Balance EUR10k-50k - dual income household - joint financial goals",
        "Balance EUR10k-50k - recent bonus deposit - lump sum received",
        "Balance EUR10k-50k - monthly standing order to savings - disciplined",
        "Balance EUR10k-50k - credit card cleared in full every month",
        "Balance EUR10k-50k - recently adopted digital channels - mobile-first",
        "Balance EUR10k-50k - approaching EUR50k - upgrade candidate",
    ]),
    ("Core savers", range(1000020, 1000030), [
        "Balance EUR3k-10k - steady growth quarter-on-quarter - active saver",
        "Balance EUR3k-10k - saving for home deposit - 2-year horizon",
        "Balance EUR3k-10k - regular pension contributor - long-term planner",
        "Balance EUR3k-10k - three or more direct debits - organised finances",
        "Balance EUR3k-10k - savings and current account active - split banking",
        "Balance EUR3k-10k - auto-transfer to savings monthly - automated habit",
        "Balance EUR3k-10k - emergency fund builder - no revolving credit",
        "Balance EUR3k-10k - moderate app usage - prefers self-service",
        "Balance EUR3k-10k - balance near EUR10k ceiling - cross-sell ready",
        "Balance EUR3k-10k - recent competitor switcher - acquisition win",
    ]),
    ("Low balance", range(1000030, 1000040), [
        "Balance <EUR1k - frequent low-value transactions - fee-sensitive",
        "Balance <EUR1k - student account - high long-term lifetime value",
        "Balance <EUR1k - seasonal income worker - irregular deposit pattern",
        "Balance <EUR1k - occasional overdraft user - repayment behaviour tracked",
        "Balance <EUR1k - recently opened - low initial deposit - nurture phase",
        "Balance <EUR1k - multiple micro-transactions daily - digital wallet user",
        "Balance <EUR1k - digital wallet primary payment method",
        "Balance <EUR1k - month-end balance spike - payroll-dependent rhythm",
        "Balance <EUR1k - credit product holder - active balance management",
        "Balance <EUR1k - recovery phase - previously overdrawn - improving",
    ]),
    ("New clients", range(1000040, 1000050), [
        "Account <6 months - acquired via referral - high conversion potential",
        "Account <6 months - acquired via online campaign - digital onboarding",
        "Account <6 months - branch walk-in acquisition - needs digital nudge",
        "Account <6 months - initial deposit >EUR5k - strong financial start",
        "Account <6 months - initial deposit <EUR500 - cautious early adopter",
        "Account <6 months - direct debit already set up - commitment signal",
        "Account <6 months - no additional products yet - onboarding pending",
        "Account <6 months - app activated - digital-first engagement",
        "Account <6 months - contacted support early - proactive relationship",
        "Account <6 months - second product added - fast engager - upsell ready",
    ]),
    ("Multi-product", range(1000050, 1000060), [
        ">3 products - mortgage, savings, card and insurance bundle",
        ">3 products - 4 active products - no obvious cross-sell gap",
        ">3 products - 5 products - premium tier eligibility confirmed",
        ">3 products - savings bond, pension and current account",
        ">3 products - business and personal accounts - dual relationship",
        ">3 products - travel card and FX account both active",
        ">3 products - auto-loan, credit card and savings - diversified",
        ">3 products - recent product addition - growing engagement trend",
        ">3 products - product renewal due within 90 days - retention alert",
        ">3 products - super-engaged - 6 or more products active",
    ]),
    ("Mortgage", range(1000060, 1000070), [
        "Active mortgage - overpaying - early repayment risk flag",
        "Active mortgage - fixed rate ending in 6 months - retention priority",
        "Active mortgage - first-time buyer - 90% LTV - high support need",
        "Active mortgage - buy-to-let investor - single rental property",
        "Active mortgage - remortgage candidate - rate review due",
        "Active mortgage - home mover - bridging finance in place",
        "Active mortgage - joint account holders - couple financial profile",
        "Active mortgage - high equity - potential equity release opportunity",
        "Active mortgage - payment holiday history - monitored account",
        "Active mortgage - multi-property portfolio - landlord segment",
    ]),
    ("CC revolvers", range(1000070, 1000080), [
        "High CC utilisation - balance transfer target - paying high APR",
        "High CC utilisation - consistent minimum payer - debt concern flag",
        "High CC utilisation - recent credit limit increase - risk review",
        "High CC utilisation - multiple cards - consolidation loan candidate",
        "High CC utilisation - recent missed payment - early delinquency risk",
        "High CC utilisation - reward card heavy spender - loyalty programme",
        "High CC utilisation - utilisation >80% - financial stress indicator",
        "High CC utilisation - improving repayment trend last 3 months",
        "High CC utilisation - travel spend dominant - frequent flyer profile",
        "High CC utilisation - recent cash advance - liquidity pressure signal",
    ]),
    ("Digital-only", range(1000080, 1000090), [
        "No branch visits - app daily active user - push notifications on",
        "No branch visits - web-only access - no mobile app installed",
        "No branch visits - instant transfer power user - real-time banking",
        "No branch visits - open banking integrations active - fintech user",
        "No branch visits - chatbot support preference - low human contact",
        "No branch visits - digital mortgage application in progress",
        "No branch visits - card controls via app - high security awareness",
        "No branch visits - fully paperless - all communications digital",
        "No branch visits - biometric login enabled - security-conscious",
        "No branch visits - digital savings pots active - goal-based saving",
    ]),
    ("Seniors 65+", range(1000090, 1000100), [
        "Age 65+ - pension income primary inflow - stable spending pattern",
        "Age 65+ - joint account with spouse - shared financial management",
        "Age 65+ - health-related spending increase - care planning phase",
        "Age 65+ - branch-dependent - low digital adoption - assisted service",
        "Age 65+ - recent legacy transfer received - wealth redistribution",
        "Age 65+ - annuity income - regular scheduled withdrawals",
        "Age 65+ - preference for gilts and bonds - low-risk investment",
        "Age 65+ - long tenure 20yr+ - highest loyalty tier",
        "Age 65+ - power-of-attorney on file - assisted account management",
        "Age 65+ - recently bereaved - account restructuring in progress",
    ]),
    ("Young adults", range(1000100, 1000110), [
        "Age 18-25 - student loan recipient - tight monthly budget",
        "Age 18-25 - first employment - salary under EUR20k - growing potential",
        "Age 18-25 - gig economy worker - irregular income - variable balance",
        "Age 18-25 - high app engagement - digital native - mobile-first",
        "Age 18-25 - rent payer - no savings yet - financial education need",
        "Age 18-25 - first credit card holder - low limit - habit forming",
        "Age 18-25 - living at home - saving for independence - goal-driven",
        "Age 18-25 - international student - FX needs - overseas transfers",
        "Age 18-25 - apprentice - expected monthly salary growth",
        "Age 18-25 - recent graduate - career starter - high lifetime value",
    ]),
    ("SME/Business", range(1000110, 1000120), [
        "Business account - sole trader - annual revenue under EUR100k",
        "Business account - limited company - 2 to 5 employees - growing",
        "Business account - seasonal business - cash-flow peaks in summer",
        "Business account - retail merchant - card terminal and POS user",
        "Business account - professional services - regular invoice cycle",
        "Business account - import/export - frequent FX transactions",
        "Business account - franchise operator - multi-site presence",
        "Business account - startup under 1 year old - high support need",
        "Business account - established SME 10yr+ - expansion phase",
        "Business account - e-commerce seller - high daily transaction volume",
    ]),
    ("Payroll", range(1000120, 1000130), [
        "Salary domiciliated - public sector employee - guaranteed stable income",
        "Salary domiciliated - private sector mid-income - steady career",
        "Salary domiciliated - high earner over EUR80k - private banking adjacent",
        "Salary domiciliated - part-time salary - supplementary income sources",
        "Salary domiciliated - recently switched - new salary mandate obtained",
        "Salary domiciliated - commission-based - variable monthly income peaks",
        "Salary domiciliated - dual salary household - both partners banking here",
        "Salary domiciliated - salary plus rental income - diversified inflows",
        "Salary domiciliated - regular overseas salary transfer - expat profile",
        "Salary domiciliated - long-term salary customer 10yr+ - loyal anchor",
    ]),
    ("Inactive", range(1000130, 1000140), [
        "Dormant 180+ days - zero transactions - fee waiver at risk",
        "Dormant - last standing order lapsed - no replacement activity",
        "Dormant - balance over EUR5k retained - passive win-back opportunity",
        "Dormant - likely moved abroad - account untouched - contact stale",
        "Dormant - reason unknown - win-back campaign target",
        "Dormant - previous complaint recorded - relationship repair needed",
        "Dormant - email unreachable - contact data requires refresh",
        "Dormant - seasonal pattern - typically reactivates in Q4",
        "Dormant - fee exemption removal flagged - action required",
        "Dormant - win-back attempt made - no response yet - escalate",
    ]),
    ("Churn-risk", range(1000140, 1000150), [
        "Churn risk - cancelled insurance product last month - early warning",
        "Churn risk - balance declining trend over 6 months - outflow pattern",
        "Churn risk - competitor account detected via FX activity",
        "Churn risk - NPS detractor - recent poor service experience",
        "Churn risk - repeated support complaints - dissatisfied customer",
        "Churn risk - card fees triggered - close-account request expected",
        "Churn risk - direct debits migrating to competitor - critical signal",
        "Churn risk - digital engagement dropping - disengagement trend",
        "Churn risk - salary no longer domiciliated - anchor product lost",
        "Churn risk - account closure requested - win-back window open now",
    ]),
]

# Flat dicts: segment_id -> short label / full description
SEGMENT_LABELS: Dict[str, str] = {}
SEGMENT_DESCRIPTIONS: Dict[str, str] = {}
for _label, _rng, _descs in _SEGMENT_ARCHETYPES:
    for _offset, _i in enumerate(_rng):
        _sid = f"{_i:07d}"
        SEGMENT_LABELS[_sid] = _label
        SEGMENT_DESCRIPTIONS[_sid] = _descs[_offset]


def generate_dummy_dataset(
    n_rows: int = 5000,
    min_users: int = 200,
    max_users: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    n_users = np.random.randint(min_users, max_users + 1)
    communications = ["day1", "day5", "day7", "day31", "day61", "day90", "day120"]
    segment_pool = [f"{i:07d}" for i in range(1000000, 1000150)]
    base_date = datetime(2024, 1, 1)

    users = [f"user_{i}" for i in range(n_users)]
    user_segments: Dict[str, List[str]] = {
        user: random.sample(segment_pool, np.random.randint(1, 5)) for user in users
    }
    user_base_balance: Dict[str, float] = {
        user: max(0.0, float(np.random.normal(3000.0, 1500.0))) for user in users
    }
    user_base_accounts: Dict[str, int] = {
        user: int(np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])) for user in users
    }

    rows = []
    for _ in range(n_rows):
        user = random.choice(users)
        communication = random.choice(communications)
        contact_flag = int(np.random.choice([0, 1], p=[0.3, 0.7]))
        start_date = base_date + timedelta(days=int(np.random.randint(0, 180)))
        end_date = start_date + timedelta(days=7)
        start_balance = max(0.0, float(np.random.normal(user_base_balance[user], 500.0)))
        start_accounts = int(user_base_accounts[user])

        if contact_flag == 1:
            balance_change = float(np.random.normal(250.0, 350.0))
            account_change = int(np.random.choice([0, 1], p=[0.75, 0.25]))
        else:
            balance_change = float(np.random.normal(50.0, 200.0))
            account_change = int(np.random.choice([0, 1], p=[0.95, 0.05]))

        end_balance = max(0.0, start_balance + balance_change)
        end_accounts = max(0, start_accounts + account_change)

        rows.append({
            "Communication":  communication,
            "alpha_key":      user,
            "Contact_flag":   contact_flag,
            "start_date":     start_date,
            "end_date":       end_date,
            "start_balance":  round(start_balance, 2),
            "end_balance":    round(end_balance, 2),
            "start_accounts": int(start_accounts),
            "end_accounts":   int(end_accounts),
            "nsegments":      user_segments[user],
        })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df_dummy = generate_dummy_dataset(n_rows=25000, min_users=200, max_users=500, seed=42)
    df_dummy.to_csv("dummy_segment_data.csv", index=False)
    print("dummy_segment_data.csv written:", len(df_dummy), "rows")

    desc_df = pd.DataFrame([
        {"nsegment": sid, "label": SEGMENT_LABELS[sid], "description": SEGMENT_DESCRIPTIONS[sid]}
        for sid in sorted(SEGMENT_LABELS)
    ])
    desc_df.to_csv("segment_descriptions.csv", index=False)
    print("segment_descriptions.csv written:", len(desc_df), "rows")
    print(df_dummy.head())
