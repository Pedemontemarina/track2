"""
Gestione degli esperimenti su disco: ogni run (controllore, ampiezza di
corrente, ...) viene salvato in una sua cartella con parametri, metriche e
serie temporale, cosi' la campagna si puo' ricaricare e confrontare senza
rilanciare le simulazioni.

Layout per run_id "PID_amp0.20":
    results/
        PID_amp0.20/
            params.json     # parametri usati per il run (controller, amplitude, ...)
            metrics.json    # metriche calcolate (pct_in_window, rmse, ...)
            timeseries.csv  # serie temporale completa (t, x, y, psi, u, v, r, tau_*, ...)
        summary.csv         # una riga per run, tutti i parametri + tutte le metriche
"""

import json
from pathlib import Path

import pandas as pd


def run_id(controller_type, current_amplitude, profile='step'):
    return f'{controller_type}_{profile}_amp{current_amplitude:.2f}'


def save_run(results_dir, rid, df, metrics, params):
    run_dir = Path(results_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(run_dir / 'timeseries.csv', index=False)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (run_dir / 'params.json').write_text(json.dumps(params, indent=2))


def load_run(results_dir, rid):
    run_dir = Path(results_dir) / rid
    df = pd.read_csv(run_dir / 'timeseries.csv')
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    params = json.loads((run_dir / 'params.json').read_text())
    return df, metrics, params


def list_runs(results_dir):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(p.name for p in results_dir.iterdir()
                  if p.is_dir() and (p / 'metrics.json').exists())


def build_summary(results_dir):
    rows = []
    for rid in list_runs(results_dir):
        _, metrics, params = load_run(results_dir, rid)
        rows.append({'run_id': rid, **params, **metrics})
    df = pd.DataFrame(rows)
    if len(df):
        Path(results_dir, 'summary.csv').write_text(df.to_csv(index=False))
    return df
