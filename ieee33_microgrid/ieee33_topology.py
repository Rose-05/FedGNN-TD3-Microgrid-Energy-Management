"""
ieee33_topology.py
──────────────────
IEEE 33-bus radial distribution system data.
Parsed from IEEE_33.xlsx (Dolatabadi et al., IEEE Trans. Power Syst. 2020).

Bus loads, branch impedances, generator buses, and 24-h temporal profiles
are all read directly from the Excel file — no hard-coded bus data.
"""

import re
import numpy as np
import pandas as pd
import pathlib


def load_ieee33(fpath: str) -> dict:
    """
    Parse IEEE_33.xlsx and return a dict of system data.

    Returns
    -------
    dict with keys:
        bus_df      pd.DataFrame  — bus number, type, Pd_MW, Qd_MVAR
        branch_df   pd.DataFrame  — from_bus, to_bus, R (Ohm), X (Ohm)
        gen_df      pd.DataFrame  — generator bus, Pmax_MW
        demand_24h  np.ndarray    — 24-h demand multiplier (p.u.)
        res_24h     np.ndarray    — 24-h RES capacity factor (p.u.)
        mg_bus      dict          — microgrid_id -> list of bus numbers
        tie_lines   list          — (from_bus, to_bus) pairs for tie-lines
        total_load_mw float       — total system load in MW
        base_mva    float
        base_kv     float
    """
    fpath = pathlib.Path(fpath).expanduser().resolve()
    if not fpath.is_file():
        raise FileNotFoundError(f"IEEE_33.xlsx not found: {fpath}")

    xl = pd.read_excel(fpath, sheet_name='case33', header=None)

    # Bus data (rows 23-55)
    raw = xl.iloc[23:56].reset_index(drop=True)
    raw.columns = range(raw.shape[1])
    bus_df = raw[[1, 2, 3, 4]].copy()
    bus_df.columns = ['bus', 'type', 'Pd_MW', 'Qd_MVAR']
    bus_df['bus']   = pd.to_numeric(bus_df['bus'],   errors='coerce')
    bus_df['Pd_MW'] = pd.to_numeric(bus_df['Pd_MW'], errors='coerce')
    bus_df = bus_df[bus_df['bus'].notna()].copy()
    bus_df['bus'] = bus_df['bus'].astype(int)
    bus_df = bus_df.reset_index(drop=True)

    # Branch data (rows 70-104)
    br = xl.iloc[70:105].reset_index(drop=True)
    br.columns = range(br.shape[1])
    branch_df = br[[1, 2, 3, 4, 5]].copy()
    branch_df.columns = ['branch', 'from_bus', 'to_bus', 'R', 'X']
    for c in branch_df.columns:
        branch_df[c] = pd.to_numeric(branch_df[c], errors='coerce')
    branch_df = branch_df.dropna().reset_index(drop=True)

    # Generator data (rows 61-65)
    gd = xl.iloc[61:66].reset_index(drop=True)
    gd.columns = range(gd.shape[1])
    gen_df = gd[[1, 2]].copy()
    gen_df.columns = ['bus', 'Pmax_MW']
    gen_df['bus']     = pd.to_numeric(gen_df['bus'],     errors='coerce')
    gen_df['Pmax_MW'] = pd.to_numeric(gen_df['Pmax_MW'], errors='coerce')
    gen_df = gen_df.dropna().reset_index(drop=True)

    def _parse_profile(cell_value) -> np.ndarray:
        nums = re.findall(r'[\d.]+', str(cell_value))
        return np.array([float(x) for x in nums], dtype=np.float32)

    demand_24h = _parse_profile(xl.iloc[112, 1])
    res_24h    = _parse_profile(xl.iloc[113, 1])

    mg_bus = {
        0: list(range(1,  7)),
        1: list(range(7,  12)),
        2: list(range(12, 18)),
        3: list(range(18, 23)),
        4: list(range(23, 28)),
        5: list(range(28, 34)),
    }
    tie_lines = [(8, 21), (12, 22), (25, 29)]
    total_load_mw = float(bus_df['Pd_MW'].sum())

    return dict(
        bus_df=bus_df, branch_df=branch_df, gen_df=gen_df,
        demand_24h=demand_24h, res_24h=res_24h,
        mg_bus=mg_bus, tie_lines=tie_lines,
        total_load_mw=total_load_mw, base_mva=100.0, base_kv=12.66,
    )


def get_bus_load_dict(ieee33: dict) -> dict:
    """Return {bus_number: Pd_MW} from parsed IEEE 33-bus data."""
    return ieee33['bus_df'].set_index('bus')['Pd_MW'].to_dict()


def get_mg_adjacency(n_mg: int = 6) -> np.ndarray:
    """
    Return (n_mg x n_mg) binary adjacency matrix derived from tie-lines.
    """
    adj = np.zeros((n_mg, n_mg), dtype=np.float32)
    mg_bus = {
        0: set(range(1, 7)),   1: set(range(7, 12)),  2: set(range(12, 18)),
        3: set(range(18, 23)), 4: set(range(23, 28)), 5: set(range(28, 34)),
    }
    ties = [(8, 21), (12, 22), (25, 29)]
    for (u, v) in ties:
        for i in range(n_mg):
            for j in range(n_mg):
                if i != j and u in mg_bus[i] and v in mg_bus[j]:
                    adj[i, j] = adj[j, i] = 1.0
    return adj


if __name__ == '__main__':
    import sys
    fpath = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/IEEE_33.xlsx'
    data = load_ieee33(fpath)
    print(f"Buses     : {len(data['bus_df'])}")
    print(f"Branches  : {len(data['branch_df'])}")
    print(f"Generators: {len(data['gen_df'])}")
    print(f"Total load: {data['total_load_mw']:.4f} MW")
    print(f"Demand profile (24h): {data['demand_24h']}")
    print(f"RES profile    (24h): {data['res_24h']}")
    print(f"\nAdjacency:\n{get_mg_adjacency()}")
