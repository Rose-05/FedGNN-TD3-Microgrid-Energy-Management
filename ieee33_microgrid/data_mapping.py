"""
data_mapping.py
───────────────
Maps measurements from Ausgrid and AEMO datasets onto the IEEE 33-bus
microgrid partition. Produces unified per-MG profiles consumed by the
environment and GNN.
"""

import numpy as np
import pandas as pd


def ausgrid_to_mg_profiles(ausgrid: dict, ieee33: dict, n_mg: int = 6) -> dict:
    """
    Map Ausgrid per-customer solar + load measurements onto MG profiles.

    Steps:
      1. Assign customers_per_mg customers per MG (sorted by ID).
      2. Compute annual-average daily profile per MG (48 x 30-min slots).
      3. Scale to MW (x customers_per_mg, / 1000).
      4. Apply IEEE 33-bus demand (load) and RES (solar) temporal profiles.
    """
    gg, gc, tcols = ausgrid['gg'], ausgrid['gc'], ausgrid['tcols']
    mg_groups = ausgrid['mg_groups']
    n_cust = len(mg_groups[0])

    d48 = np.repeat(ieee33['demand_24h'], 2).astype(np.float32)
    r48 = np.repeat(ieee33['res_24h'],    2).astype(np.float32)

    solar_raw = np.array([
        gg[gg['Customer'].isin(g)][tcols].mean().values
        for g in mg_groups], dtype=np.float32)

    load_raw = np.array([
        gc[gc['Customer'].isin(g)][tcols].mean().values
        for g in mg_groups], dtype=np.float32)

    gen_cap = np.array([
        gg[gg['Customer'].isin(g)]['Generator Capacity'].dropna().mean()
        for g in mg_groups], dtype=np.float32)

    solar_mw = (solar_raw * n_cust / 1000.0) * r48[np.newaxis, :]
    load_mw  = (load_raw  * n_cust / 1000.0) * d48[np.newaxis, :]

    return dict(solar_mw=solar_mw, load_mw=load_mw,
                gen_cap_kw=gen_cap, mg_groups=mg_groups)


def aemo_to_mg_wind(aemo: dict, n_mg: int = 6,
                    wind_cap_per_mg: float = 1.8,
                    rng: np.random.Generator = None) -> dict:
    """
    Derive per-MG wind profiles and electricity price proxy from AEMO data.

    NEM-wide wind total is scaled to per-MG distributed wind capacity (~1.8 MW).
    Price proxy: price_t = clip(300 - 0.22 * wind_NEM_t, 40, 300)  [$/MWh]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    wind_avg = aemo['wind_avg']
    price_48 = aemo['price_48']

    wscale  = wind_cap_per_mg / (wind_avg.max() + 1e-9)
    wind_cv = float(aemo['aemo_df']['wind_NEM'].std() /
                    (aemo['aemo_df']['wind_NEM'].mean() + 1e-9))

    wind_mw = np.array([
        np.clip(wind_avg * wscale * (1.0 + wind_cv*(rng.random()-0.5)),
                0.05*wind_cap_per_mg, wind_cap_per_mg)
        for _ in range(n_mg)
    ], dtype=np.float32)

    return dict(wind_mw=wind_mw, price_48=price_48)


def build_profiles(ausgrid: dict, aemo: dict, ieee33: dict,
                   n_mg: int = 6,
                   rng: np.random.Generator = None) -> dict:
    """Build complete integrated profiles dict consumed by MicrogridEnv."""
    aus_map  = ausgrid_to_mg_profiles(ausgrid, ieee33, n_mg)
    aemo_map = aemo_to_mg_wind(aemo, n_mg, rng=rng)

    price_buy  = aemo_map['price_48'] / 1000.0
    price_sell = price_buy * 0.55

    mg_bus_load = {
        mg: float(ieee33['bus_df'].set_index('bus')
                  .reindex(buses)['Pd_MW'].sum())
        for mg, buses in ieee33['mg_bus'].items()
    }

    return dict(
        solar_mw=aus_map['solar_mw'],
        load_mw=aus_map['load_mw'],
        wind_mw=aemo_map['wind_mw'],
        price_buy=price_buy,
        price_sell=price_sell,
        demand_48=np.repeat(ieee33['demand_24h'], 2).astype(np.float32),
        res_48=np.repeat(ieee33['res_24h'], 2).astype(np.float32),
        t_half=np.arange(48, dtype=np.float32) * 0.5,
        mg_bus_load=mg_bus_load,
        _ausgrid=ausgrid, _aemo=aemo, _ieee33=ieee33,
    )
