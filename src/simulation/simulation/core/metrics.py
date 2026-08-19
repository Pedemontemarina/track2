"""
Metriche per confrontare PID e SMC sul task di net-distance-keeping.
Tutte calcolate dalla sola serie temporale (distanza, comandi tau).
"""

import numpy as np


def compute_metrics(t, distance, tau_u, tau_v, tau_r, d_ref,
                     window=(0.70, 1.50), t_onset=5.0, settle_band=0.05):
    t = np.asarray(t)
    distance = np.asarray(distance)
    dt = np.gradient(t)

    lo, hi = window
    in_window = (distance >= lo) & (distance <= hi)
    pct_in_window = 100.0 * np.trapz(in_window.astype(float), t) / (t[-1] - t[0])

    rmse = float(np.sqrt(np.mean((distance - d_ref) ** 2)))

    min_distance = float(np.min(distance))

    # Tempo di assestamento dopo t_onset: primo istante da cui la distanza
    # resta stabilmente entro +-settle_band*d_ref attorno a d_ref fino alla
    # fine della simulazione (definizione "non abbandona piu' la banda").
    band = settle_band * d_ref
    mask_post = t >= t_onset
    settled = np.abs(distance - d_ref) <= band
    settling_time = np.nan
    idx_post = np.where(mask_post)[0]
    for i in idx_post:
        if np.all(settled[i:]):
            settling_time = float(t[i] - t_onset)
            break

    control_effort = float(np.trapz(tau_u ** 2 + tau_v ** 2 + tau_r ** 2, t))

    return dict(
        pct_in_window=pct_in_window,
        rmse=rmse,
        min_distance=min_distance,
        settling_time=settling_time,
        control_effort=control_effort,
    )
