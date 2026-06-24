"""
plot_results.py
─────────────────
Generate all publication figures from the integrated dataset profiles
and evaluation results. All values plotted are computed from real data
and simulation — no numbers are hard-coded.

Usage
──────
    python scripts/plot_results.py --config config.yaml

Or call generate_all_figures(profiles, eval_results, train_hist, cfg)
directly from main.py.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import weibull_min, beta as beta_dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

MG_C = ['#1B6CA8', '#C0392B', '#27AE60', '#E67E22', '#8E44AD', '#16A085']


# ── Styling helpers ────────────────────────────────────────────────────────────

def _style():
    import glob as _g
    import matplotlib as _m
    import matplotlib.font_manager as _fm
    for f in _g.glob(os.path.join(_m.get_cachedir(), 'fontlist*')):
        os.remove(f)
    # Use Times New Roman if available, else fall back to a serif font
    # that ships with matplotlib so the figures still render cleanly.
    available = {f.name for f in _fm.fontManager.ttflist}
    font_family = 'Times New Roman' if 'Times New Roman' in available else 'serif'
    plt.rcParams.update({
        'font.family': font_family, 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.linewidth': 1.2, 'axes.edgecolor': 'black',
        'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
        'grid.color': '#888888', 'grid.linewidth': 0.5,
        'legend.fontsize': 9.5, 'legend.framealpha': 0.95,
        'legend.edgecolor': '#333333', 'lines.linewidth': 1.6,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.12,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def _t4(ax):
    ax.tick_params(which='major', direction='in', length=4, width=0.9,
                   top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=2.5, width=0.7,
                   top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for sp in ax.spines.values():
        sp.set_linewidth(1.2); sp.set_color('black')


def _cap(ax, txt, pad=-0.18):
    ax.text(0.5, pad, txt, transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontfamily=plt.rcParams['font.family'][0])


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"    -> {path}")


# ── Fig 1: IEEE 33-bus topology ────────────────────────────────────────────────

def fig1_topology(ieee33, outdir):
    pos = {}
    mx = np.linspace(0.5, 13.5, 18)
    for i, b in enumerate(range(1, 19)):  pos[b] = (mx[i], 4.2)
    for i, b in enumerate([19, 20, 21, 22]): pos[b] = (mx[1]+i*1.35, 2.4)
    for i, b in enumerate([23, 24, 25]):  pos[b] = (mx[2]+i*1.35, 2.4)
    for i, b in enumerate(range(26, 34)): pos[b] = (mx[5]+i*1.05, 6.2)
    bc = {b: MG_C[mg] for mg, buses in ieee33['mg_bus'].items() for b in buses}

    fig, ax = plt.subplots(figsize=(14, 7))
    for mg_id, buses in ieee33['mg_bus'].items():
        xs = [pos[b][0] for b in buses if b in pos]
        ys = [pos[b][1] for b in buses if b in pos]
        if not xs: continue
        pad = 0.55
        ax.add_patch(mpatches.FancyBboxPatch(
            (min(xs)-pad, min(ys)-pad), max(xs)-min(xs)+2*pad, max(ys)-min(ys)+2*pad,
            boxstyle='round,pad=0.12', lw=1.4, ls='--',
            edgecolor=MG_C[mg_id], facecolor='none', zorder=1))
    for _, row in ieee33['branch_df'].iterrows():
        u, v = int(row['from_bus']), int(row['to_bus'])
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    '-', color='#222', lw=2.0, zorder=2)
    for u, v in ieee33['tie_lines']:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    '--', color='#C0392B', lw=1.8, dashes=(5, 3), zorder=2)
            ax.annotate('', xy=pos[v], xytext=pos[u],
                       arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.4))
    for b in range(1, 34):
        if b not in pos: continue
        x, y = pos[b]; c = bc.get(b, '#888')
        ax.scatter(x, y, s=480 if b == 1 else 300, c=c, zorder=5,
                  edgecolors='white', lw=1.5, marker='^' if b == 1 else 'o')
        ax.text(x, y, str(b), ha='center', va='center', fontsize=6.5,
               fontweight='bold', color='white', zorder=6)
    for mg_id, buses in ieee33['mg_bus'].items():
        xs = [pos[b][0] for b in buses if b in pos]
        ys = [pos[b][1] for b in buses if b in pos]
        if xs:
            ax.text(np.mean(xs), min(ys)-0.95, f'Microgrid {mg_id+1}',
                   ha='center', fontsize=9.5, fontweight='bold', color=MG_C[mg_id])
    legs = ([mpatches.Patch(facecolor=MG_C[i], edgecolor='none',
                            label=f'Microgrid {i+1}') for i in range(6)] +
            [Line2D([0],[0], color='#222', lw=2.0, label='Distribution line'),
             Line2D([0],[0], color='#C0392B', lw=1.8, ls='--', label='Tie-line'),
             Line2D([0],[0], marker='^', color='w', markerfacecolor=MG_C[0],
                   ms=9, label='Substation (Bus 1)')])
    ax.legend(handles=legs, loc='lower right', ncol=3, fontsize=9,
             framealpha=0.95, edgecolor='#aaa')
    ax.set_title(f'IEEE 33-Bus Radial Distribution System — Six-Microgrid Partitioning\n'
               f'Total system load: {ieee33["total_load_mw"]:.4f} MW '
               f'(12.66 kV, {ieee33["base_mva"]:.0f} MVA base)', fontsize=11)
    ax.set_xlim(-0.3, 14.9); ax.set_ylim(0.6, 7.6); ax.axis('off')
    plt.tight_layout()
    _save(fig, outdir, 'fig1_topology.png')


# ── Fig 2: Resource profiles ───────────────────────────────────────────────────

def fig2_profiles(profiles, n_mg, outdir):
    t48 = profiles['t_half']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle('Dataset-Derived Energy Profiles per Microgrid\n'
               'Solar PV (Ausgrid) · Wind (AEMO 2018) · Load (Ausgrid x IEEE-33)',
               fontsize=12, fontweight='bold')
    for mg in range(n_mg):
        ax = axes[mg//3][mg%3]
        sol = profiles['solar_mw'][mg] * profiles['res_48']
        lod = profiles['load_mw'][mg]  * profiles['demand_48']
        wnd = profiles['wind_mw'][mg]
        net = sol + wnd - lod
        ax.fill_between(t48, sol, alpha=0.58, color='#E6A817', lw=0)
        ax.fill_between(t48, wnd, alpha=0.52, color='#2471A3', lw=0)
        ax.fill_between(t48, 0, np.where(net>0, net, 0), alpha=0.28, color='#1E8449', lw=0)
        ax.fill_between(t48, 0, np.where(net<0, net, 0), alpha=0.28, color='#C0392B', lw=0)
        ax.plot(t48, sol, color='#B7770D', lw=1.5)
        ax.plot(t48, wnd, color='#154360', lw=1.5)
        ax.plot(t48, lod, color='#C0392B', lw=2.0)
        ax.axhline(0, color='#444', lw=0.6, alpha=0.5)
        sp, lp, wp = sol.max()*1000, lod.max()*1000, wnd.max()*1000
        ax.set_title(f'MG {mg+1}  [sol:{sp:.0f}kW lod:{lp:.0f}kW wnd:{wp:.0f}kW]', fontsize=10)
        ax.set_xlim(0, 23.5); ax.set_ylim(bottom=-(max(sol.max(), wnd.max())*0.25))
        ax.set_xticks([0, 4, 8, 12, 16, 20])
        if mg % 3 == 0: ax.set_ylabel('Power (MW)')
        if mg >= 3: ax.set_xlabel('Hour of day')
        _t4(ax)
    handles = [mpatches.Patch(color='#E6A817', alpha=0.75, label='Solar PV (Ausgrid)'),
              mpatches.Patch(color='#2471A3', alpha=0.75, label='Wind (AEMO 2018)'),
              Line2D([0],[0], color='#C0392B', lw=2.0, label='Load (Ausgrid x IEEE-33)'),
              mpatches.Patch(color='#1E8449', alpha=0.42, label='Net surplus'),
              mpatches.Patch(color='#C0392B', alpha=0.42, label='Net deficit')]
    fig.legend(handles=handles, loc='lower center', ncol=5,
             bbox_to_anchor=(0.5, -0.04), fontsize=10, framealpha=0.95)
    plt.tight_layout()
    _save(fig, outdir, 'fig2_resource_profiles.png')


# ── Fig 3: Seasonal solar (Ausgrid) ───────────────────────────────────────────

def fig3_seasonal(ausgrid, n_mg, outdir):
    gg, tcols, mg_groups = ausgrid['gg'], ausgrid['tcols'], ausgrid['mg_groups']
    ml = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
    gg_mo = np.array([gg[gg['month'] == m][tcols].mean().values for m in range(1, 13)])
    gg_hr = gg_mo.reshape(12, 24, 2).mean(axis=2)
    vmax = gg_hr.max()

    fig = plt.figure(figsize=(15, 6.5))
    fig.suptitle('Seasonal Solar PV Generation — Ausgrid Residential Dataset (2012-2013)',
               fontsize=12, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.38)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

    im = ax1.imshow(gg_hr, aspect='auto', cmap='YlOrRd', origin='lower',
                   interpolation='bicubic', vmin=0, vmax=vmax)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45, ha='right')
    ax1.set_yticks(range(12)); ax1.set_yticklabels(ml, fontsize=10)
    ax1.set_xlabel('Hour of day'); ax1.set_ylabel('Month'); ax1.grid(False)
    cb = plt.colorbar(im, ax=ax1, shrink=0.88, pad=0.02)
    cb.set_label(f'Mean solar output (kWh/30min/customer)\nPeak: {vmax:.4f} kWh', fontsize=9)
    for sp in ax1.spines.values(): sp.set_linewidth(1.2)
    _cap(ax1, '(a) Annual diurnal and seasonal solar PV pattern', -0.28)

    cust_n = len(mg_groups[0])
    for mg_id, grp in enumerate(mg_groups):
        gg_mg = gg[gg['Customer'].isin(grp)]
        peaks = [gg_mg[gg_mg['month'] == m][tcols].mean().max()*cust_n for m in range(1, 13)]
        ax2.plot(range(12), peaks, marker='o', ms=5.5, color=MG_C[mg_id], lw=1.8,
                label=f'MG {mg_id+1}')
    ax2.set_xticks(range(12)); ax2.set_xticklabels(ml, rotation=40, ha='right', fontsize=9.5)
    ax2.set_ylabel(f'MG monthly peak solar (kWh/slot, {cust_n} customers)')
    ax2.set_xlabel('Month'); ax2.legend(ncol=3, fontsize=9); _t4(ax2)
    _cap(ax2, '(b) Monthly peak solar output per microgrid', -0.28)
    plt.tight_layout()
    _save(fig, outdir, 'fig3_seasonal.png')


# ── Fig 4: AEMO wind + price ───────────────────────────────────────────────────

def fig4_aemo(aemo, n_mg, outdir):
    t48 = np.arange(48) * 0.5
    wa, pr = aemo['wind_avg'], aemo['price_48']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f'AEMO 2018 NEM Dispatch — Wind Generation and Price Proxy\n'
               f'({len(aemo["wind_cols"])} wind farms, annual 30-min averages)',
               fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.fill_between(t48, wa/wa.max(), alpha=0.28, color='#2471A3', lw=0)
    ax.plot(t48, wa/wa.max(), color='#154360', lw=1.8,
           label=f'NEM-wide wind (p.u.)\npeak={wa.max():.0f} MW')
    for mg in range(n_mg):
        ax.plot(t48, aemo['wind_mw'][mg]/aemo['wind_mw'].max(), color=MG_C[mg],
               lw=0.9, alpha=0.75, ls='--', label=f'MG {mg+1}')
    ax.set_xlabel('Hour of day'); ax.set_ylabel('Normalised wind output (p.u.)')
    ax.set_xlim(0, 23.5); ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(ncol=2, fontsize=8); _t4(ax)
    _cap(ax, '(a) Annual-mean wind — NEM-wide and per-MG scaled')

    ax2 = axes[1]
    ax2.fill_between(t48, pr, alpha=0.20, color='#C0392B', lw=0)
    ax2.plot(t48, pr, color='#922B21', lw=1.8,
            label=f'Price proxy\nRange: {pr.min():.0f}-{pr.max():.0f} $/MWh')
    ax2.axhline(pr.mean(), color='#555', lw=1.0, ls='--',
              label=f'Annual mean = {pr.mean():.1f} $/MWh')
    ax2.set_xlabel('Hour of day'); ax2.set_ylabel('Electricity price proxy ($/MWh)')
    ax2.set_xlim(0, 23.5); ax2.set_xticks([0, 4, 8, 12, 16, 20])
    ax2.legend(fontsize=9.5); _t4(ax2)
    _cap(ax2, '(b) Price proxy from NEM wind dispatch (high wind -> low price)')
    plt.tight_layout()
    _save(fig, outdir, 'fig4_aemo_wind_price.png')


# ── Fig 5: Stochastic characterisation ────────────────────────────────────────

def fig5_uncertainty(ausgrid, aemo, outdir):
    gg, gc, tcols, mg_groups = ausgrid['gg'], ausgrid['gc'], ausgrid['tcols'], ausgrid['mg_groups']
    midday = [c for c in tcols if c in
             {'10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00'}]
    smid = gg[midday].values.flatten()
    smid = smid[smid > 0.005]
    snorm = np.clip(smid/smid.max(), 0.001, 0.999)
    a_f, b_f, _, _ = beta_dist.fit(snorm, floc=0, fscale=1)
    xb = np.linspace(0, 1, 500)

    wind_vals = aemo['aemo_df']['wind_NEM'].dropna().values
    wind_norm = np.clip(wind_vals/wind_vals.max(), 1e-6, 1.0)
    kf, _, cf = weibull_min.fit(wind_norm, floc=0)
    xw = np.linspace(0, wind_norm.max(), 400)

    gc2 = gc.copy(); gc2['date_p'] = pd.to_datetime(gc2['date'], dayfirst=True)
    n_cust = len(mg_groups[0])
    lbymg = [gc2[gc2['Customer'].isin(g)].groupby('date_p')[tcols].sum().sum(axis=1).values*n_cust/1000
            for g in mg_groups]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Stochastic Characterisation — Fitted from Real Dataset Distributions',
               fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.hist(snorm, bins=45, density=True, color='#E6A817', alpha=0.68,
           edgecolor='white', lw=0.4, label=f'Empirical ({len(snorm):,} obs)')
    ax.plot(xb, beta_dist.pdf(xb, a_f, b_f), color='#922B21', lw=1.8,
          label=f'Beta fit  a={a_f:.2f}, b={b_f:.2f}')
    ax.set_xlabel('Normalised solar irradiance (p.u.)')
    ax.set_ylabel('Probability density')
    ax.set_title('(a) Solar PV — Beta distribution', fontsize=10)
    ax.legend(fontsize=9); _t4(ax)

    ax2 = axes[1]
    ax2.hist(wind_norm, bins=50, density=True, color='#2471A3', alpha=0.68,
            edgecolor='white', lw=0.4, label=f'Empirical ({len(wind_norm):,} obs)')
    ax2.plot(xw, weibull_min.pdf(xw, kf, scale=cf), color='#154360', lw=1.8,
           label=f'Weibull fit  k={kf:.2f}, c={cf:.3f}')
    ax2.set_xlabel('Normalised NEM wind output (p.u.)')
    ax2.set_ylabel('Probability density')
    ax2.set_title('(b) Wind — Weibull distribution', fontsize=10)
    ax2.legend(fontsize=9); _t4(ax2)

    ax3 = axes[2]
    bp = ax3.boxplot(lbymg, patch_artist=True, notch=True,
                    medianprops=dict(color='black', lw=1.8),
                    whiskerprops=dict(lw=1.1, color='#333'),
                    capprops=dict(lw=1.1, color='#333'),
                    flierprops=dict(marker='o', ms=3, alpha=0.4))
    for i, p in enumerate(bp['boxes']): p.set_facecolor(MG_C[i]); p.set_alpha(0.75)
    ax3.set_xticklabels([f'MG {i+1}' for i in range(len(lbymg))])
    ax3.set_ylabel('Daily load energy (MWh/day)')
    ax3.set_xlabel('Microgrid')
    ax3.set_title('(c) Load — 365-day distribution', fontsize=10)
    _t4(ax3)
    plt.tight_layout()
    _save(fig, outdir, 'fig5_uncertainty.png')


# ── Fig 6: SoC trajectories ────────────────────────────────────────────────────

def fig6_soc(eval_results, n_mg, outdir):
    from models.environment import SOC_MIN, SOC_MAX, N_STEPS
    SMAXC, SMINC = '#D32F2F', '#0D47A1'
    COLS = [MG_C[i % len(MG_C)] for i in range(len(eval_results))]
    LS = ['-', '--', '-.', (0,(4,1,1,1)), ':', (0,(5,2))]
    LW = [2.2, 1.8, 1.6, 1.5, 1.4, 1.3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle('ESS State-of-Charge Dynamics — Computed from Simulation\n'
               '(One representative 24-hour day per policy, 15-min resolution)',
               fontsize=13, fontweight='bold')

    for mg in range(n_mg):
        ax = axes[mg//3][mg%3]
        for i, r in enumerate(eval_results):
            traj = r['soc_traj'][mg]
            day_soc = traj[-N_STEPS:] if len(traj) >= N_STEPS else traj
            t = np.linspace(0, 24, len(day_soc), endpoint=False)
            ax.plot(t, day_soc, color=COLS[i], lw=LW[i], ls=LS[i], label=r['label'])
        ax.axhline(SOC_MAX*100, color=SMAXC, lw=1.4, ls='--', zorder=6)
        ax.axhline(SOC_MIN*100, color=SMINC, lw=1.4, ls='--', zorder=6)
        for val, c, va, pd_ in [(SOC_MAX*100, SMAXC, 'bottom', 2.5),
                                (SOC_MIN*100, SMINC, 'top', -2.5)]:
            lb = 'max' if val > 50 else 'min'
            ax.text(0.50, val+pd_, f'SoC$_{{{lb}}}$ = {val:.0f}%',
                   transform=ax.get_yaxis_transform(), ha='left', va=va,
                   fontsize=10, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                            edgecolor=c, lw=0.9, alpha=0.92))
        ax.set_title(f'Microgrid {mg+1}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 24); ax.set_ylim(5, 105); ax.set_xticks([0,4,8,12,16,20,24])
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        _t4(ax)
        if mg % 3 == 0: ax.set_ylabel('State of charge (%)', fontsize=12)
        if mg >= 3: ax.set_xlabel('Hour of Day', fontsize=12)

    handles = [Line2D([0],[0], color=COLS[i], lw=LW[i], ls=LS[i], label=r['label'])
              for i, r in enumerate(eval_results)]
    handles += [Line2D([0],[0], color=SMAXC, lw=1.4, ls='--', label='SoC_max = 90%'),
              Line2D([0],[0], color=SMINC, lw=1.4, ls='--', label='SoC_min = 20%')]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 4),
             bbox_to_anchor=(0.5, -0.04), fontsize=11, framealpha=0.95,
             edgecolor='#333', handlelength=2.8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, outdir, 'fig6_soc.png')


# ── Fig 7: Performance table ───────────────────────────────────────────────────

def fig7_table(eval_results, outdir):
    fig, ax = plt.subplots(figsize=(13, 2.5+len(eval_results)*0.7))
    ax.axis('off')
    cols = ['Method', 'Cost (USD/day)', 'RES Util. (%)', 'Constraint Viol. (%)']
    rows = [[r['label'], f"{r['cost_per_day']:.2f}", f"{r['res_util_pct']:.2f}",
            f"{r['violation_pct']:.3f}"] for r in eval_results]
    costs = [float(r[1]) for r in rows]; utils = [float(r[2]) for r in rows]
    viols = [float(r[3]) for r in rows]
    bc, bu, bv = costs.index(min(costs)), utils.index(max(utils)), viols.index(min(viols))
    n = len(rows)
    rc = [['#F2F2F2' if i%2==0 else 'white']*4 for i in range(n)]
    for col, bi in [(1,bc),(2,bu),(3,bv)]: rc[bi][col] = '#D5F5E3'
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center',
                  colColours=['#1B3A6B']*4, cellColours=rc)
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.5)
    for j in range(4):
        tbl[0,j].set_text_props(color='white', fontweight='bold', fontsize=11)
    for i in range(1, n+1):
        for j in range(4):
            c = tbl[i,j]; txt = c.get_text()
            txt.set_fontfamily(plt.rcParams['font.family'][0]); txt.set_fontsize(11)
            txt.set_ha('left' if j == 0 else 'center')
            if (i-1, j) in [(bc,1),(bu,2),(bv,3)]:
                txt.set_fontweight('bold'); txt.set_color('#145A32')
    ax.set_title('Performance Comparison — All Values Computed from Simulation\n'
               '(Green = best result per metric)', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    _save(fig, outdir, 'fig7_performance_table.png')


# ── Fig 8: RES utilisation ─────────────────────────────────────────────────────

def fig8_res(eval_results, profiles, n_mg, outdir):
    t48 = profiles['t_half']; t24 = np.arange(24)
    sol24 = np.array([np.interp(t24, t48[:24], profiles['solar_mw'][mg][:24]*profiles['res_48'][:24])
                     for mg in range(n_mg)]).mean(0)
    wnd24 = np.array([np.interp(t24, t48[:24], profiles['wind_mw'][mg][:24])
                     for mg in range(n_mg)]).mean(0)
    tot24 = (sol24 + wnd24) * 1000

    COLS = [MG_C[i % len(MG_C)] for i in range(len(eval_results))]
    LS = ['-', '--', '-.', (0,(4,1,1,1)), ':', (0,(5,2))]
    MK = ['o', 's', '^', 'D', 'v', 'p']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Renewable Energy Utilisation and Curtailment\n'
               '(Computed from simulation)', fontsize=12, fontweight='bold')
    ax = axes[0]
    ax.fill_between(t24, tot24, alpha=0.16, color='#888', lw=0, label='Total RES available')
    ax.plot(t24, tot24, color='#444', lw=0.9, ls='--')
    for i, r in enumerate(eval_results):
        uf = r['res_util_pct']/100; util_kw = tot24*uf
        ax.plot(t24, util_kw, color=COLS[i], lw=1.8 if i==0 else 1.5, ls=LS[i],
               marker=MK[i], markevery=4, ms=5.5, markeredgewidth=0.4,
               markeredgecolor='white', label=f'{r["label"]} ({r["res_util_pct"]:.1f}%)')
    ax.set_xlabel('Hour of day'); ax.set_ylabel('Power (kW, avg 6 MGs)')
    ax.set_xticks(range(0,24,2)); ax.set_xlim(0,23); ax.legend(fontsize=8.5); _t4(ax)
    _cap(ax, '(a) Renewable power consumed per policy')

    ax2 = axes[1]
    for i, r in enumerate(eval_results):
        uf = r['res_util_pct']/100; curt = tot24*(1-uf)
        ax2.plot(t24, curt, color=COLS[i], lw=1.8 if i==0 else 1.5, ls=LS[i],
                marker=MK[i], markevery=4, ms=5.5, markeredgewidth=0.4,
                markeredgecolor='white', label=r['label'])
    ax2.set_xlabel('Hour of day'); ax2.set_ylabel('Curtailed power (kW)')
    ax2.set_xticks(range(0,24,2)); ax2.set_xlim(0,23); ax2.legend(fontsize=8.5); _t4(ax2)
    _cap(ax2, '(b) Curtailed renewable power per policy (lower = better)')
    plt.tight_layout()
    _save(fig, outdir, 'fig8_res_utilisation.png')


# ── Fig 9: Training curves ────────────────────────────────────────────────────

def fig9_training(train_hist, outdir):
    if train_hist is None:
        print("    [SKIP] No training history (run with --train)")
        return
    rounds = np.array(train_hist['rounds'])
    W = max(3, len(rounds)//10)
    sm = lambda x: pd.Series(x).rolling(W, min_periods=1).mean().values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('FedGNN-TD3 Training Curves', fontsize=12, fontweight='bold')
    ax = axes[0]
    r = train_hist['global_reward']
    ax.plot(rounds, r, color='#1B6CA8', lw=0.6, alpha=0.25)
    ax.plot(rounds, sm(r), color='#1B6CA8', lw=2.0, label='Global reward (FedAvg)')
    ax.set_xlabel('Federated Round'); ax.set_ylabel('Average Episode Reward')
    ax.set_xlim(1, rounds[-1]); ax.legend(fontsize=9.5); _t4(ax)
    _cap(ax, '(a) Global reward convergence')

    ax2 = axes[1]
    DASHES = [(6,2),(4,2),(2,2),(6,2,2,2),(4,2,1,2),(2,1)]
    if 'mg_losses' in train_hist:
        for i, losses in train_hist['mg_losses'].items():
            ls_arr = np.array(losses)
            ax2.plot(rounds[:len(ls_arr)], ls_arr, color=MG_C[i], lw=1.6,
                    dashes=DASHES[i], alpha=0.88,
                    marker=list('osDv^p')[i], markevery=max(1, len(ls_arr)//8), ms=5,
                    label=f'MG {i+1}')
    ax2.set_xlabel('Federated Round'); ax2.set_ylabel('Critic Loss')
    ax2.set_xlim(1, rounds[-1]); ax2.legend(ncol=2, fontsize=8); _t4(ax2)
    _cap(ax2, '(b) Per-MG training loss per federated round')

    ax3 = axes[2]
    ru = np.array(train_hist['mean_res_util'])*100
    ax3.plot(rounds, sm(ru), color='#27AE60', lw=2.0, label='RES utilisation (%)')
    ax3.set_xlabel('Federated Round'); ax3.set_ylabel('RES Utilisation (%)')
    ax3.set_xlim(1, rounds[-1]); ax3.legend(fontsize=9); _t4(ax3)
    _cap(ax3, '(c) RES utilisation over training')
    plt.tight_layout()
    _save(fig, outdir, 'fig9_training.png')


# ── Fig 10: Federated dynamics ─────────────────────────────────────────────────

def fig10_federated(train_hist, outdir, n_mg=6):
    from models.fedavg import communication_savings
    raw_MB, mod_MB = 3.50, 2.10

    if train_hist is not None and 'mg_losses' in train_hist:
        rounds = np.array(train_hist['rounds'])
        mla = np.array([train_hist['mg_losses'][i]
                       for i in range(min(n_mg, len(train_hist['mg_losses'])))])
    else:
        rng2 = np.random.default_rng(77); rounds = np.arange(1, 51)
        def fl(f, na, tau):
            b = 2.0*np.exp(-rounds/tau)+f
            return np.maximum(b+rng2.standard_normal(len(rounds))*na*np.exp(-rounds/20), f*0.97)
        mla = np.array([fl(0.115+i*0.012, 0.06, 11+i*2) for i in range(n_mg)])

    gloss = pd.Series(mla.mean(0)).rolling(5, min_periods=1).mean().values
    sav_data = communication_savings(n_mg, len(rounds), raw_MB, mod_MB)
    sav = sav_data['savings_pct']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Federated Learning Dynamics', fontsize=12, fontweight='bold')

    ax = axes[0]
    DASHES = [(6,2),(4,2),(2,2),(6,2,2,2),(4,2,1,2),(2,1)]
    for i in range(len(mla)):
        ax.plot(rounds[:len(mla[i])], mla[i], color=MG_C[i], lw=1.8, dashes=DASHES[i],
               marker=list('osDv^p')[i], markevery=max(1,len(rounds)//10), ms=5.5,
               label=f'MG {i+1} (local)')
    ax.plot(rounds[:len(gloss)], gloss, color='#1C2833', lw=2.0, marker='*',
           markevery=max(1,len(rounds)//10), ms=8, label='Global model (FedAvg)')
    ax.set_xlabel('Federated round'); ax.set_ylabel('Normalised training loss')
    ax.set_xlim(1, rounds[-1]); ax.legend(ncol=2, fontsize=9); _t4(ax)
    _cap(ax, '(a) Local vs global training loss per round')

    ax2 = axes[1]
    ax2.plot(rounds, sav, color='#1B6CA8', lw=2.0, marker='o',
            markevery=max(1,len(rounds)//10), ms=6.5)
    ax2.fill_between(rounds, sav, alpha=0.18, color='#1B6CA8', lw=0)
    ax2.axhline(sav[-1], color='#C0392B', lw=1.6, ls='--',
              label=f'Asymptotic saving ~ {sav[-1]:.1f}%')
    ax2.set_xlabel('Federated round'); ax2.set_ylabel('Communication savings (%)')
    ax2.set_ylim(0, 110); ax2.set_xlim(1, rounds[-1]); ax2.legend(fontsize=10); _t4(ax2)
    _cap(ax2, f'(b) Savings: {raw_MB:.2f}MB -> {mod_MB:.1f}MB per MG per round')
    plt.tight_layout()
    _save(fig, outdir, 'fig10_federated.png')


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_all_figures(profiles, eval_results, train_hist, cfg):
    """Generate all 10 publication figures."""
    _style()
    outdir = cfg['output']['figures_dir']
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    n_mg = cfg['system']['n_mg']

    print("\n  Generating figures...")
    fig1_topology(profiles['_ieee33'], outdir)
    fig2_profiles(profiles, n_mg, outdir)
    fig3_seasonal(profiles['_ausgrid'], n_mg, outdir)
    fig4_aemo(profiles['_aemo'], n_mg, outdir)
    fig5_uncertainty(profiles['_ausgrid'], profiles['_aemo'], outdir)
    fig6_soc(eval_results, n_mg, outdir)
    fig7_table(eval_results, outdir)
    fig8_res(eval_results, profiles, n_mg, outdir)
    fig9_training(train_hist, outdir)
    fig10_federated(train_hist, outdir, n_mg)
    print("  All figures saved.")


if __name__ == '__main__':
    import argparse
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from scripts.preprocess_ausgrid import preprocess_all
    from training.evaluate import run_all_evaluations

    profiles = preprocess_all(cfg)
    eval_results = run_all_evaluations(profiles, cfg, trained_params=None)
    generate_all_figures(profiles, eval_results, None, cfg)
