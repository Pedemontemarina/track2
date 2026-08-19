"""
Motore di simulazione headless (nessun ROS2) per la campagna di esperimenti
PID vs SMC x ampiezza di corrente, pensato per essere chiamato dal notebook.

Task: rete nel piano x=0, ROV che parte lontano (x0) e deve mantenere
d(t) = x(t) in [70,150] cm attorno a d_ref, mentre y e psi sono tenuti a
zero (station-keeping laterale/di prua), come discusso nel capitolo
Control Methods esteso a 3 assi indipendenti (modello disaccoppiato).
"""

import numpy as np
import pandas as pd

from .dynamics import BlueROV2Dynamics
from .controllers import PIDAxis, SMCAxis
from .currents import CurrentProfile


# Guadagni di default, tarati in acqua calma (current_amplitude=0) per
# ottenere un assestamento senza overshoot pericoloso e senza oscillazioni
# sostenute. Vedi notebook 01_calibration per il procedimento.
DEFAULT_GAINS = {
    'PID': {
        'surge': dict(kp=18.0, ki=2.5, kd=14.0),
        'sway': dict(kp=12.0, ki=1.5, kd=10.0),
        'yaw': dict(kp=3.0, ki=0.2, kd=1.5),
    },
    'SMC': {
        'surge': dict(lam=1.0, k=18.0, phi=0.08),
        'sway': dict(lam=1.0, k=10.0, phi=0.08),
        'yaw': dict(lam=1.0, k=1.0, phi=0.08),
    },
}


def _build_controllers(controller_type, gains, dyn):
    g = gains[controller_type]
    if controller_type == 'PID':
        return dict(
            surge=PIDAxis(**g['surge'], integral_limit=5.0),
            sway=PIDAxis(**g['sway'], integral_limit=5.0),
            yaw=PIDAxis(**g['yaw'], integral_limit=2.0, angular=True),
        )
    elif controller_type == 'SMC':
        return dict(
            surge=SMCAxis(dyn.I_u, dyn.X_u, dyn.X_uu, **g['surge']),
            sway=SMCAxis(dyn.I_v, dyn.Y_v, dyn.Y_vv, **g['sway']),
            yaw=SMCAxis(dyn.I_z, dyn.N_r, dyn.N_rr, **g['yaw'], angular=True),
        )
    else:
        raise ValueError(f'unknown controller_type {controller_type}')


def run_experiment(controller_type, current_amplitude, current_direction=0.0,
                    profile='step', duration=60.0, dt=0.01, t_onset=5.0,
                    ramp_duration=5.0, d_ref=1.10, x0=2.5, y0=0.0, psi0=0.0,
                    window=(0.70, 1.50), gains=None, tau_limit=30.0,
                    dyn_params=None):
    """Esegue un run e ritorna (DataFrame serie temporali, dict metriche)."""
    from .metrics import compute_metrics

    gains = gains or DEFAULT_GAINS
    dyn = BlueROV2Dynamics(**(dyn_params or {}))
    ctrl = _build_controllers(controller_type, gains, dyn)
    current = CurrentProfile(current_amplitude, current_direction, profile,
                              t_onset, ramp_duration)

    n_steps = int(duration / dt)
    state = np.array([x0, y0, psi0, 0.0, 0.0, 0.0], dtype=float)

    rows = []
    for i in range(n_steps):
        t = i * dt
        x, y, psi, u, v, r = state

        tau_u = ctrl['surge'].compute(d_ref, x, u, dt)
        tau_v = ctrl['sway'].compute(0.0, y, v, dt)
        tau_r = ctrl['yaw'].compute(0.0, psi, r, dt)
        tau = np.clip([tau_u, tau_v, tau_r], -tau_limit, tau_limit)

        cur = current(t)
        rows.append((t, x, y, psi, u, v, r, *tau, cur[0], cur[1]))

        state = dyn.step(state, tau, cur, dt)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f'Simulazione divergente a t={t:.2f}s (controller={controller_type}, '
                f'current_amplitude={current_amplitude})')

    df = pd.DataFrame(rows, columns=[
        't', 'x', 'y', 'psi', 'u', 'v', 'r',
        'tau_u', 'tau_v', 'tau_r', 'current_x', 'current_y',
    ])
    df['distance'] = df['x']

    metrics = compute_metrics(
        df['t'].values, df['distance'].values,
        df['tau_u'].values, df['tau_v'].values, df['tau_r'].values,
        d_ref, window=window, t_onset=t_onset,
    )
    metrics.update(dict(
        controller=controller_type, current_amplitude=current_amplitude,
        current_direction=current_direction, profile=profile,
    ))

    return df, metrics


def run_campaign(controllers, current_amplitudes, results_dir, profile='step',
                  save=True, **kwargs):
    """
    Lancia un run per ogni combinazione (controllore x ampiezza), salva
    ciascuno su disco (se save=True) e ritorna il DataFrame riassuntivo
    (una riga per run, parametri + metriche).
    """
    from .storage import run_id, save_run, build_summary

    for controller_type in controllers:
        for amp in current_amplitudes:
            df, metrics = run_experiment(controller_type, amp, profile=profile, **kwargs)
            if save:
                rid = run_id(controller_type, amp, profile)
                params = dict(controller=controller_type, current_amplitude=amp,
                              profile=profile, **{k: v for k, v in kwargs.items()
                                                   if k not in ('gains',)})
                save_run(results_dir, rid, df, metrics, params)

    return build_summary(results_dir) if save else None
