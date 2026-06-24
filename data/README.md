# Datasets

Three datasets are required. Download each and place in `data/raw/`.

---

## 1. Ausgrid Solar Home Electricity Data 2012–2013

**Download:** https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data

**File to place:** `data/raw/2012-2013_Solar_home_electricity_data_v2.csv`

**What it contains:**
- 300 residential customers in Sydney, Australia
- Period: July 2012 – June 2013 (365 days)
- 48 half-hour time slots per day
- `Consumption Category`: `GG` = solar generation, `GC` = general load consumption
- `Generator Capacity` column: installed solar PV capacity per customer (1–9.99 kW)

**How it is used in FedGNN-TD3:**
- Customers are partitioned into 6 groups of 50 → one group per microgrid
- `GG` column averages → solar PV generation profile per MG (MW, 48 slots)
- `GC` column averages → residential load demand profile per MG (MW, 48 slots)
- Temporal shape is further modulated by the IEEE 33-bus 24h demand multiplier

---

## 2. AEMO 2018 National Electricity Market (NEM) Dispatch Data

**Download:** https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem

**File to place:** `data/raw/aemo_2018.csv`

**What it contains:**
- Full-year 2018 dispatch data at 5-minute resolution (105,120 rows)
- 33 wind farm columns (suffix `WF`): e.g. ARWF1, CAPTL_WF, HALLWF1 …
- 28 solar/CSPV farm columns (suffix `SF` or `CSPV`): e.g. BNGSF1, CSPVPS1 …
- NEM-wide wind capacity: 8 – 2587 MW; solar: 0 – 1180 MW

**How it is used in FedGNN-TD3:**
- Wind farms aggregated to NEM total, resampled to 30-min → scaled to per-MG capacity (~1.8 MW)
- Solar farms used for cross-validation of Ausgrid PV profiles
- Price proxy: `price = clip(300 − 0.22 × wind_MW, 40, 300)` $/MWh
  (merit-order approximation: high wind → low wholesale price)

---

## 3. IEEE 33-Bus Enhanced Benchmark (IEEE_33.xlsx)

**Citation:** S. H. Dolatabadi, M. Ghorbanian, P. Siano, and N. D. Hatziargyriou, "An Enhanced IEEE 33 Bus Benchmark Test System for Distribution System Studies," *IEEE Trans. Power Syst.*, 2020.

**File to place:** `data/raw/IEEE_33.xlsx`

**What it contains (sheet: `case33`):**
- Rows 23–55: Bus data — bus number, type (PQ/PV/ref), Pd (MW), Qd (MVAr)
- Rows 61–65: Generator data — buses 1, 18, 22, 25, 33; Pmax = 4.0 / 0.2 MW
- Rows 70–104: Branch data — 35 branches, R and X in Ohm
- Row 112: 24-hour demand multiplier profile (p.u.)
- Row 113: 24-hour RES capacity factor profile (p.u.)

**System parameters:**
- 33 buses, base voltage 12.66 kV, base power 100 MVA
- Total system load: 4.0865 MW
- 6 microgrid partitions with 3 switchable tie-lines: (8,21), (12,22), (25,29)

---

## Expected directory structure after downloading

```
data/
├── raw/
│   ├── 2012-2013_Solar_home_electricity_data_v2.csv   ← Ausgrid
│   ├── aemo_2018.csv                                   ← AEMO
│   └── IEEE_33.xlsx                                    ← IEEE 33-bus
└── processed/                                          ← auto-generated
```
