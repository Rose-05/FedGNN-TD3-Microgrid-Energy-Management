"""
preprocess_ausgrid.py
──────────────────────
Load and preprocess all three datasets (Ausgrid, AEMO, IEEE_33.xlsx)
and merge them into the unified per-microgrid profiles dict consumed
by the training and evaluation pipelines.

Despite the filename (kept for clarity since Ausgrid is the primary
load/solar source), this module loads and integrates ALL THREE datasets.
"""

import pathlib
import re
import numpy as np
import pandas as pd


def _resolve_path(hint, default_rel_path):
    """Resolve a dataset path: use hint if given, else config default."""
    p = pathlib.Path(hint if hint else default_rel_path).expanduser().resolve()
    return str(p)


def load_ausgrid(fpath: str, n_mg: int, customers_per_mg: int) -> dict:
    """Load Ausgrid residential solar/load CSV."""
    print(f"  Ausgrid     : {fpath}")
    df = pd.read_csv(fpath, skiprows=1, low_memory=False)
    df['date_parsed'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date_parsed'].dt.month
    tcols = [c for c in df.columns if ':' in c][:48]

    gg = df[df['Consumption Category'] == 'GG'].copy()
    gc = df[df['Consumption Category'] == 'GC'].copy()

    customers = sorted(df['Customer'].unique())
    if len(customers) < n_mg * customers_per_mg:
        raise ValueError(
            f"Not enough customers ({len(customers)}) for "
            f"{n_mg} x {customers_per_mg} = {n_mg*customers_per_mg}")

    mg_groups = [customers[i*customers_per_mg:(i+1)*customers_per_mg]
                 for i in range(n_mg)]

    solar_raw = np.array([gg[gg['Customer'].isin(g)][tcols].mean().values
                          for g in mg_groups], dtype=np.float32)
    load_raw  = np.array([gc[gc['Customer'].isin(g)][tcols].mean().values
                          for g in mg_groups], dtype=np.float32)
    gen_cap   = np.array([gg[gg['Customer'].isin(g)]['Generator Capacity']
                          .dropna().mean() for g in mg_groups], dtype=np.float32)

    solar_mw = solar_raw * customers_per_mg / 1000.0
    load_mw  = load_raw  * customers_per_mg / 1000.0

    print(f"    Customers: {df['Customer'].nunique()} | "
          f"Period: {df['date_parsed'].min().date()} - "
          f"{df['date_parsed'].max().date()}")
    for i in range(n_mg):
        print(f"    MG{i+1}: solar_peak={solar_mw[i].max():.4f} MW  "
              f"load_peak={load_mw[i].max():.4f} MW")

    return dict(solar_mw=solar_mw, load_mw=load_mw, gen_cap_per_mg=gen_cap,
                mg_groups=mg_groups, tcols=tcols, gg=gg, gc=gc, df=df)


def load_aemo(fpath: str, n_mg: int, rng: np.random.Generator) -> dict:
    """Load AEMO 2018 5-minute dispatch CSV."""
    print(f"  AEMO 2018   : {fpath}")
    aemo = pd.read_csv(fpath, low_memory=False)
    aemo['ts']   = pd.to_datetime(aemo['timestamp'], utc=True)
    aemo['slot'] = (aemo['ts'].dt.hour * 60 + aemo['ts'].dt.minute) // 30

    wind_cols  = [c for c in aemo.columns if 'WF' in c]
    solar_cols = [c for c in aemo.columns if ('SF' in c or 'CSPV' in c)]

    aemo[wind_cols]  = aemo[wind_cols].apply(pd.to_numeric,  errors='coerce')
    aemo[solar_cols] = aemo[solar_cols].apply(pd.to_numeric, errors='coerce')
    aemo['wind_NEM']  = aemo[wind_cols].sum(axis=1,  min_count=1).clip(lower=0)
    aemo['solar_NEM'] = aemo[solar_cols].sum(axis=1, min_count=1).clip(lower=0)

    slot_avg = aemo.groupby('slot')[['wind_NEM', 'solar_NEM']].mean()
    wind_avg = slot_avg['wind_NEM'].values.astype(np.float32)

    price_48 = (300.0 - 0.22 * wind_avg).clip(40, 300).astype(np.float32)

    wind_cap   = 1.8
    wind_scale = wind_cap / (wind_avg.max() + 1e-9)
    wind_cv    = float(aemo.groupby('slot')['wind_NEM'].std().mean() /
                       (wind_avg.mean() + 1e-9))

    wind_mw = np.array([
        np.clip(wind_avg * wind_scale * (1.0 + wind_cv*(rng.random()-0.5)),
                0.05*wind_cap, wind_cap)
        for _ in range(n_mg)
    ], dtype=np.float32)

    print(f"    Wind farms: {len(wind_cols)} | Solar farms: {len(solar_cols)}")
    print(f"    NEM wind: {wind_avg.min():.0f}-{wind_avg.max():.0f} MW -> "
          f"per-MG: {wind_mw.min():.3f}-{wind_mw.max():.3f} MW")
    print(f"    Price proxy: {price_48.min():.1f}-{price_48.max():.1f} $/MWh")

    return dict(wind_mw=wind_mw, wind_avg=wind_avg,
                solar_avg=slot_avg['solar_NEM'].values.astype(np.float32),
                price_48=price_48, wind_cols=wind_cols, solar_cols=solar_cols,
                aemo_df=aemo[['ts', 'slot', 'wind_NEM', 'solar_NEM']].copy())


def load_ieee33(fpath: str) -> dict:
    """Load IEEE_33.xlsx — bus data, branches, 24h profiles."""
    from ieee33_microgrid.ieee33_topology import load_ieee33 as _load
    return _load(fpath)


def preprocess_all(cfg: dict) -> dict:
    """
    Load and integrate all three datasets according to config.yaml paths.
    Returns the unified `profiles` dict used by training/evaluate modules.
    """
    from ieee33_microgrid.data_mapping import build_profiles

    n_mg = cfg['system']['n_mg']
    cust_per_mg = cfg['system']['customers_per_mg']
    seed = cfg['training']['seed']
    rng  = np.random.default_rng(seed)

    ausgrid_path = _resolve_path(None, cfg['data']['ausgrid'])
    aemo_path    = _resolve_path(None, cfg['data']['aemo'])
    ieee_path    = _resolve_path(None, cfg['data']['ieee33'])

    for p, name in [(ausgrid_path, 'Ausgrid'), (aemo_path, 'AEMO'),
                    (ieee_path, 'IEEE_33')]:
        if not pathlib.Path(p).is_file():
            raise FileNotFoundError(
                f"{name} dataset not found at: {p}\n"
                f"Update the path in config.yaml under data:{name.lower()}, "
                f"or place the file as documented in data/README.md")

    ieee33  = load_ieee33(ieee_path)
    ausgrid = load_ausgrid(ausgrid_path, n_mg, cust_per_mg)
    aemo    = load_aemo(aemo_path, n_mg, rng)

    profiles = build_profiles(ausgrid, aemo, ieee33, n_mg=n_mg, rng=rng)

    print("\n  Integration summary:")
    for mg in range(n_mg):
        sp = profiles['solar_mw'][mg].max()
        lp = profiles['load_mw'][mg].max()
        wp = profiles['wind_mw'][mg].max()
        print(f"    MG{mg+1}: solar={sp:.4f} MW  load={lp:.4f} MW  wind={wp:.4f} MW")

    return profiles


if __name__ == '__main__':
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    profiles = preprocess_all(cfg)
    print("\nPreprocessing complete.")
