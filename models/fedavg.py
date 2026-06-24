"""
fedavg.py
─────────
Federated Averaging (FedAvg) aggregation for FedGNN-TD3.

Implements Equations 14-15 from the paper:
    theta_global^{r+1} = sum_i (n_i / sum_j n_j) * theta_i^r       Eq. 14
    theta_i^{r+1} <- theta_global^{r+1},  for all i                Eq. 15

Privacy guarantee: only model parameters (actor + critic weights) are
transmitted to the server — never raw operational data.
"""

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False


def federated_average(local_params: list, n_samples: list) -> dict:
    """
    Weighted FedAvg aggregation (Eq. 14).

    Parameters
    ----------
    local_params : list of dicts, each {'actor': state_dict, 'critic': state_dict}
    n_samples    : list of int — number of local training samples per MG
                   (used as aggregation weights, proportional to data volume)

    Returns
    -------
    dict {'actor': aggregated_state_dict, 'critic': aggregated_state_dict}
    """
    if not TORCH or not local_params:
        return {}

    total = sum(n_samples)
    weights = [n / total for n in n_samples]

    global_params = {}
    for net_name in ('actor', 'critic'):
        global_params[net_name] = {}
        for key in local_params[0][net_name].keys():
            global_params[net_name][key] = sum(
                w * p[net_name][key].float()
                for w, p in zip(weights, local_params)
            )
    return global_params


def communication_savings(n_mg: int, fed_rounds: int,
                          raw_data_mb_per_mg: float = 3.50,
                          model_mb_per_mg: float = 2.10) -> dict:
    """
    Estimate communication overhead reduction vs centralised raw-data sharing.

    Default values reflect the actual data volume of one day of Ausgrid
    measurements (raw_data_mb_per_mg ~ 50 customers x 365 days x 48 slots x
    4 bytes / 1e6) versus the size of GNN + TD3 model parameters
    (model_mb_per_mg).

    Returns
    -------
    dict with cumulative raw/model MB transmitted and percentage savings
    over `fed_rounds` rounds.
    """
    import numpy as np
    rounds = np.arange(1, fed_rounds + 1)
    cum_raw   = rounds * n_mg * raw_data_mb_per_mg
    cum_model = rounds * n_mg * model_mb_per_mg
    savings_pct = (cum_raw - cum_model) / cum_raw * 100
    return dict(rounds=rounds, cum_raw_mb=cum_raw,
                cum_model_mb=cum_model, savings_pct=savings_pct)


if __name__ == '__main__':
    sav = communication_savings(n_mg=6, fed_rounds=50)
    print(f"Round 1   savings: {sav['savings_pct'][0]:.1f}%")
    print(f"Round 50  savings: {sav['savings_pct'][-1]:.1f}%")
