# Power BI Setup Guide — Affinity Explorer

This guide walks you through recreating the Streamlit Affinity Explorer dashboard
in Power BI using the pre-computed data tables.

---

## 1. Generate the Data Tables

```powershell
cd "Santander segments"
.venv\Scripts\python.exe powerbi_data_prep.py
```

This creates a `powerbi/` folder with these CSVs:

| File | Description | Rows (approx) |
|------|-------------|----------------|
| `dim_segments.csv` | Segment IDs, labels, descriptions, groups | 40 |
| `dim_communications.csv` | Communication names + sort order | 7 |
| `fact_observations.csv` | Row-level exploded data | ~370K |
| `fact_segment_stats.csv` | Pre-computed stats per segment × comm | ~280 |
| `fact_combo_pairs.csv` | Segment pair combos with synergy | varies |
| `fact_combo_triples.csv` | Segment triple combos with synergy | varies |
| `fact_cooccurrence.csv` | Pairwise overlap (long format) | ~1,600 |
| `fact_journey.csv` | Journey timeline means | ~210 |

---

## 2. Import into Power BI

1. Open Power BI Desktop → **Get Data** → **Text/CSV**
2. Import each CSV from the `powerbi/` folder
3. In Power Query Editor, verify column types:
   - Percentages (`*_bal`, `*_acct`, `*_lift_*`, `*_pct*`) → **Decimal Number**
   - Counts (`n_*`, `*_customers`) → **Whole Number**
   - Dates (`start_date`, `end_date`) → **Date**
   - IDs and names → **Text**
4. Click **Close & Apply**

---

## 3. Create Relationships (Data Model)

Switch to **Model View** and create these relationships:

```
dim_segments.nsegment  ──1:*──  fact_segment_stats.nsegment
dim_segments.nsegment  ──1:*──  fact_observations.nsegment
dim_segments.nsegment  ──1:*──  fact_journey.nsegment

dim_communications.communication  ──1:*──  fact_segment_stats.communication
dim_communications.communication  ──1:*──  fact_observations.communication
dim_communications.communication  ──1:*──  fact_journey.communication

fact_cooccurrence: NO relationships (standalone matrix)
fact_combo_pairs:  NO relationships (standalone table)
fact_combo_triples: NO relationships (standalone table)
```

**Cross-filter direction**: Set all to **Single** (dimension → fact).

---

## 4. Create DAX Measures

1. Right-click in the Fields pane → **New Table** → name it `Measures`
2. Open `powerbi_measures.dax` and create each measure from that file
3. Key measures to add first:
   - `Lift Bal%`, `Lift Acct%` — single-cell lifts
   - `Lift Bal% (N-weighted)` — weighted lift across selected segments
   - `Total Treated Customers`
   - `Is Significant (Bal)` — for conditional formatting
   - `Lift Color`, `N Color` — for conditional formatting rules

---

## 5. Build Report Pages

### Page 1: Segment Explorer (≈ Streamlit "Segment Explorer" tab)

| Visual | Configuration |
|--------|--------------|
| **Slicer** | `dim_segments[group]` — filter by segment group |
| **Table** | Rows: `dim_segments[nsegment]`, `dim_segments[label]`, `dim_segments[description]` <br> Values: `[Unique Customers]` <br> Conditional formatting on Unique Customers using `N Color` |
| **Bar chart** | X: `dim_segments[group]` Y: Count of `dim_segments[nsegment]` |
| **Bar chart** | X: `dim_segments[group]` Y: `[Unique Customers]` |

### Page 2: Segment × Communication Table (≈ "Table" tab)

| Visual | Configuration |
|--------|--------------|
| **Slicer** (horizontal) | `dim_communications[communication]` |
| **Matrix** | Rows: `fact_segment_stats[nsegment]` <br> Columns: `fact_segment_stats[communication]` <br> Values: `[Lift Bal%]` <br> Conditional formatting → Background color → Rules → use `Lift Color` measure |

**Alternative**: Use a **Table** visual with `fact_segment_stats` fields directly:
- nsegment, communication, treated_mean_bal, lift_bal, lift_ci_bal, n_treated
- Apply conditional formatting: lift_bal → Background color → Color scale (Red-Yellow-Green)

### Page 3: Targeting Simulator (≈ "Targeting Simulator" tab)

| Visual | Configuration |
|--------|--------------|
| **Slicer** | `dim_segments[nsegment]` (multi-select) — user picks segments |
| **Card** | `[Total Treated Customers]` |
| **Card** | `[Lift Bal% (N-weighted)]` — headline lift for selection |
| **Card** | `[Projected Balance Increase (€)]` |
| **Matrix** | Rows: `fact_segment_stats[nsegment]` <br> Columns: `fact_segment_stats[communication]` <br> Values: `[Lift Bal%]` |
| **Table** | `fact_combo_pairs` → segments, combo_lift_bal, best_individual_lift, synergy, n_customers <br> Sort by synergy DESC |
| **Scatter** | X: `fact_combo_pairs[combo_lift_bal]` Y: `fact_combo_pairs[synergy]` <br> Size: `fact_combo_pairs[n_customers]` <br> Color: `fact_combo_pairs[synergy]` |

### Page 4: Charts (≈ "Charts" tab)

| Visual | Configuration |
|--------|--------------|
| **Slicer** | `dim_communications[communication]` (single select) |
| **Bar chart** | X: `fact_segment_stats[nsegment]` Y: `[Lift Bal%]` <br> Top N filter: top 25 by lift_bal |
| **Matrix (heatmap)** | Rows: `fact_segment_stats[nsegment]` <br> Columns: `fact_segment_stats[communication]` <br> Values: `fact_segment_stats[treated_mean_bal]` <br> Conditional formatting → Color scale |
| **Line chart** | X: `fact_journey[communication]` Y: `[Journey Bal%]` <br> Legend: `fact_journey[nsegment]` <br> Sort: `fact_journey[comm_sort_order]` |
| **Matrix (co-occurrence)** | Rows: `fact_cooccurrence[segment_a]` <br> Columns: `fact_cooccurrence[segment_b]` <br> Values: `[Overlap %]` <br> Conditional formatting → Blues color scale |

### Page 5: Data Quality (≈ "Data Quality" tab)

| Visual | Configuration |
|--------|--------------|
| **Card** | `COUNTROWS(fact_observations)` — Total records |
| **Card** | `[Unique Customers]` |
| **Card** | `[Contact Rate %]` |
| **Card** | `[Missing Balance %]` |
| **Card** | `[Multi-Segment %]` |
| **Table** | Communication × Contact% × Control% (from fact_segment_stats aggregated) |

---

## 6. Conditional Formatting

For any lift column in a Table or Matrix visual:
1. Select the column → **Conditional formatting** → **Background color**
2. Choose **Color scale**: Red (#F8696B) → Yellow (#FFEB84) → Green (#63BE7B)
3. Set Minimum = -10%, Midpoint = 0%, Maximum = +10%

For significance (muted vs strong colors):
1. Use the `Is Significant (Bal)` measure
2. Apply **Rules**: if = 1 → bold color; if = 0 → 50% transparency

For N (sample size) columns:
1. **Rules**: < 30 → Red, 30-100 → Yellow, > 100 → Green

---

## 7. Tooltips

Power BI supports **tooltip pages** for hover detail:
1. Create a new page → Page Size → Tooltip
2. Add fields: segment label, description, lift, N, projected impact
3. On your main visuals, set Tooltip → this page

To show segment labels on hover like the Streamlit app:
- Add `dim_segments[label]` and `dim_segments[description]` to the **Tooltips** bucket of each visual

---

## 8. Slicers to Replicate Sidebar Filters

| Streamlit Filter | Power BI Slicer |
|-----------------|-----------------|
| Date range | `fact_observations[start_date]` — Between slicer |
| Min-N | Visual-level filter on `n_treated >= [value]` |
| Communication checkboxes | `dim_communications[communication]` multi-select |
| Segment group | `dim_segments[group]` dropdown |

For the **min-N filter**: Use a "What-if" parameter:
1. **Modeling** → **New Parameter** → Name: "Min N", Start: 0, End: 500, Default: 30
2. Create measure: `Filtered Lift = IF([Total Treated Customers] >= [Min N Value], [Lift Bal%], BLANK())`
3. Use `Filtered Lift` instead of `Lift Bal%` in visuals

---

## What Power BI Can't Do Natively

| Feature | Streamlit | Power BI Workaround |
|---------|-----------|-------------------|
| Segment combo explorer (itertools) | Python-computed on the fly | Pre-computed tables (`fact_combo_pairs`, `fact_combo_triples`) |
| Violin / KDE plots | Plotly | Use histogram or Python visual |
| Custom HTML table with JS sorting | Full control | Use Matrix visual with conditional formatting |
| Weighted CI calculations | NumPy | Pre-computed in `fact_segment_stats` |
| Real-time parameter adjustment | Streamlit sliders | What-if parameters + calculated measures |

---

## Refreshing Data

When the source CSV changes:
1. Re-run `python powerbi_data_prep.py` to regenerate the `powerbi/` folder
2. In Power BI → **Home** → **Refresh** to pick up the new CSVs

For automatic refresh, set up a Power BI Gateway pointing to the `powerbi/` folder.
