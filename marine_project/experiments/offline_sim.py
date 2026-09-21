#!/usr/bin/env python3
"""
offline_sim.py

Simulazione OFFLINE del BlueROV2 (dinamica + PID/SMC), senza ROS2.
Stessa fisica e stessi controllori dei nodi ROS2 -- qui solo per
iterare velocemente sul tuning, prima di portare i guadagni buoni
nei nodi veri.

USO TIPICO
    Importa le funzioni in un notebook/REPL e chiama run_simulation(...)
    con i parametri che vuoi provare.
"""

import numpy as np


# =============================================================================
# PARAMETRI FISICI (Tabella I del paper -- convenzione positiva per la
# massa aggiunta, negativa per il damping. Stessa convenzione usata nei
# nodi ROS2: tenerla allineata se cambi qualcosa qui.)
# =============================================================================
PARAMS = dict(
    m_rb=11.5,
    m_a_u=5.50, m_a_v=12.70,
    iz_rigid=0.16, m_a_r=0.12,
    x_u=-4.03, x_uu=-18.18,
    y_v=-6.22, y_vv=-21.66,
    n_r=-0.07, n_rr=-1.55,
)

I_u = PARAMS['m_rb'] + PARAMS['m_a_u']
I_v = PARAMS['m_rb'] + PARAMS['m_a_v']
I_z = PARAMS['iz_rigid'] + PARAMS['m_a_r']


# =============================================================================
# DINAMICA (identica a dynamics_node._dynamics, senza ROS)
# =============================================================================
def dynamics(state, tau, current_inertial):
    """state = [x, y, psi, u, v, r] -> restituisce state_dot."""
    x, y, psi, u, v, r = state
    tau_u, tau_v, tau_r = tau
    cos_p, sin_p = np.cos(psi), np.sin(psi)

    uc_n, vc_n = current_inertial
    uc_b = cos_p * uc_n + sin_p * vc_n
    vc_b = -sin_p * uc_n + cos_p * vc_n
    u_r, v_r = u - uc_b, v - vc_b

    u_dot = (tau_u + PARAMS['x_u'] * u_r + PARAMS['x_uu'] * abs(u_r) * u_r) / I_u
    v_dot = (tau_v + PARAMS['y_v'] * v_r + PARAMS['y_vv'] * abs(v_r) * v_r) / I_v
    r_dot = (tau_r + PARAMS['n_r'] * r + PARAMS['n_rr'] * abs(r) * r) / I_z

    x_dot = u * cos_p - v * sin_p
    y_dot = u * sin_p + v * cos_p
    psi_dot = r

    return np.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot])


def rk4_step(state, tau, current, dt):
    k1 = dynamics(state, tau, current)
    k2 = dynamics(state + dt / 2 * k1, tau, current)
    k3 = dynamics(state + dt / 2 * k2, tau, current)
    k4 = dynamics(state + dt * k3, tau, current)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


# =============================================================================
# CONTROLLORI (identici alle classi AxisPID / AxisSMC dei nodi ROS2)
# =============================================================================
class AxisPID:
    def __init__(self, kp, ki, kd, tau_max, integral_max, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.tau_max, self.integral_max, self.dt = tau_max, integral_max, dt
        self.integral_e = 0.0
        self.prev_e = 0.0

    def update(self, e):
        self.integral_e = np.clip(self.integral_e + e * self.dt,
                                   -self.integral_max, self.integral_max)
        e_dot = (e - self.prev_e) / self.dt
        self.prev_e = e
        tau = self.kp * e + self.ki * self.integral_e + self.kd * e_dot
        return float(np.clip(tau, -self.tau_max, self.tau_max))


def sat(x, limit=1.0):
    return float(np.clip(x, -limit, limit))


class AxisSMC:
    def __init__(self, I, d_lin, d_quad, lam, k_switch, phi, tau_max):
        self.I, self.d_lin, self.d_quad = I, d_lin, d_quad
        self.lam, self.k_switch, self.phi, self.tau_max = lam, k_switch, phi, tau_max

    def update(self, e, vel):
        s = -vel + self.lam * e
        tau_eq = -self.d_lin * vel - self.d_quad * abs(vel) * vel - self.I * self.lam * vel
        tau_switch = self.I * self.k_switch * sat(s / self.phi)
        tau = tau_eq + tau_switch
        return float(np.clip(tau, -self.tau_max, self.tau_max))


def make_pid_controllers(dt, gains=None):
    g = gains or dict(
        kp_u=7.5, ki_u=0.65, kd_u=11.0,
        kp_v=9.5, ki_v=0.85, kd_v=20.0,
        kp_r=0.44, ki_r=0.055, kd_r=0.49,
    )
    tau_max_linear = g.get('tau_max_linear', 60.0)
    tau_max_yaw = g.get('tau_max_yaw', 5.0)
    integral_max_u = g.get('integral_max_u', 25.0)
    integral_max_v = g.get('integral_max_v', 25.0)
    integral_max_r = g.get('integral_max_r', 1.0)

    pid_u = AxisPID(g['kp_u'], g['ki_u'], g['kd_u'], tau_max_linear, integral_max_u, dt)
    pid_v = AxisPID(g['kp_v'], g['ki_v'], g['kd_v'], tau_max_linear, integral_max_v, dt)
    pid_r = AxisPID(g['kp_r'], g['ki_r'], g['kd_r'], tau_max_yaw, integral_max_r, dt)
    return pid_u, pid_v, pid_r


def make_smc_controllers(gains=None):
    g = gains or dict(
        lambda_u=0.5, k_u=3.55, phi_u=0.083,
        lambda_v=0.5, k_v=4.70, phi_v=0.09,
        lambda_r=1.0, k_r=2.0, phi_r=0.1,
    )
    # PRIMA erano scritti fissi a 30.0/5.0 dentro AxisSMC(...) -- non
    # leggevano affatto da 'gains'. Ora, stessa logica del PID:
    tau_max_linear = g.get('tau_max_linear', 60.0)
    tau_max_yaw = g.get('tau_max_yaw', 5.0)

    smc_u = AxisSMC(I_u, PARAMS['x_u'], PARAMS['x_uu'],
                    g['lambda_u'], g['k_u'], g['phi_u'], tau_max_linear)
    smc_v = AxisSMC(I_v, PARAMS['y_v'], PARAMS['y_vv'],
                    g['lambda_v'], g['k_v'], g['phi_v'], tau_max_linear)
    smc_r = AxisSMC(I_z, PARAMS['n_r'], PARAMS['n_rr'],
                    g['lambda_r'], g['k_r'], g['phi_r'], tau_max_yaw)
    return smc_u, smc_v, smc_r


# =============================================================================
# LOOP DI SIMULAZIONE
# =============================================================================
def run_simulation(controller_type, x_ref, y_ref, psi_ref,
                    current_magnitude=0.0, current_direction_deg=0.0,
                    duration=60.0, dt=0.01, control_dt=0.02, gains=None,
                    x0=0.0, y0=0.0, psi0=0.0):
    """
    controller_type: 'pid' oppure 'smc'
    Ritorna un dizionario con le serie temporali (per plottare/analizzare).
    """
    current = np.array([
        current_magnitude * np.cos(np.deg2rad(current_direction_deg)),
        current_magnitude * np.sin(np.deg2rad(current_direction_deg)),
    ])

    if controller_type == 'pid':
        ctrl_u, ctrl_v, ctrl_r = make_pid_controllers(control_dt, gains)
    elif controller_type == 'smc':
        ctrl_u, ctrl_v, ctrl_r = make_smc_controllers(gains)
    else:
        raise ValueError("controller_type deve essere 'pid' o 'smc'")

    state = np.array([x0, y0, psi0, 0.0, 0.0, 0.0])   # x, y, psi, u, v, r
    n_steps = int(duration / dt)
    control_every = max(1, int(control_dt / dt))

    history = dict(t=[], x=[], y=[], psi=[], u=[], v=[], r=[],
                    tau_u=[], tau_v=[], tau_r=[])
    tau = np.zeros(3)

    for i in range(n_steps):
        t = i * dt

        if i % control_every == 0:
            x, y, psi, u, v, r = state
            e_x = x_ref - x
            e_y = y_ref - y
            e_psi = wrap_angle(psi_ref - psi)

            if controller_type == 'pid':
                tau_u = ctrl_u.update(e_x)
                tau_v = ctrl_v.update(e_y)
                tau_r = ctrl_r.update(e_psi)
            else:
                tau_u = ctrl_u.update(e_x, u)
                tau_v = ctrl_v.update(e_y, v)
                tau_r = ctrl_r.update(e_psi, r)

            tau = np.array([tau_u, tau_v, tau_r])

        state = rk4_step(state, tau, current, dt)

        history['t'].append(t)
        history['x'].append(state[0])
        history['y'].append(state[1])
        history['psi'].append(state[2])
        history['u'].append(state[3])
        history['v'].append(state[4])
        history['r'].append(state[5])
        history['tau_u'].append(tau[0])
        history['tau_v'].append(tau[1])
        history['tau_r'].append(tau[2])

    return {k: np.array(v) for k, v in history.items()}


# =============================================================================
# METRICHE UTILI PER IL CONFRONTO PID VS SMC
# =============================================================================
def distance_metrics(hist, net_x, window=(0.70, 1.50)):
    """Percentuale di tempo dentro la finestra 70-150cm + RMSE sul target."""
    d = net_x - hist['x']
    in_window = np.logical_and(d >= window[0], d <= window[1])
    pct_in_window = 100.0 * np.mean(in_window)
    return dict(pct_in_window=pct_in_window, d=d)


def offset_metrics(hist, target, key, tolerance=0.05):
    """Percentuale di tempo entro 'tolerance' dal target, per y o psi."""
    err = np.abs(hist[key] - target)
    in_tolerance = err <= tolerance
    return 100.0 * np.mean(in_tolerance)

def full_metrics(hist, target, key, tol_frac=0.05, tail_seconds=10.0):
    t = hist['t']
    val = hist[key]

    # Se target=0, usa una tolleranza assoluta piccola invece di tol_frac*|target|
    if abs(target) < 1e-9:
        band = 0.05  # fascia assoluta di default per target=0 (es. 5cm o 0.05 rad)
    else:
        band = tol_frac * abs(target)

    outside = np.abs(val - target) > band
    if outside.any():
        last_outside_idx = np.where(outside)[0][-1]
        settling_time = None if last_outside_idx == len(t)-1 else t[last_outside_idx+1]
    else:
        settling_time = t[0]

    # Overshoot: se target=0, riporta il valore assoluto del picco, non una percentuale
    if abs(target) < 1e-9:
        overshoot = max(abs(val.max()), abs(val.min()))  # picco assoluto raggiunto
    elif target > 0:
        overshoot = max(0.0, (val.max() - target) / target * 100)
    else:
        overshoot = max(0.0, (target - val.min()) / abs(target) * 100)

    tail_mask = t >= (t[-1] - tail_seconds)
    steady_state_error = np.mean(val[tail_mask]) - target

    return dict(settling_time=settling_time, overshoot_pct=overshoot,
                steady_state_error=steady_state_error)

def chattering_metric(hist, key='tau_u', skip_transient=0.5, amplitude_threshold=0.01):
    """
    Conta i cambi di direzione al secondo, MA ignora le variazioni
    sotto amplitude_threshold (rumore numerico, non chattering vero).
    """
    mask = hist['t'] > skip_transient
    tau = hist[key][mask]
    d_tau = np.diff(tau)

    # tieni solo i punti dove la variazione è abbastanza grande da essere reale
    significant = np.abs(d_tau) > amplitude_threshold
    d_tau_significant = d_tau[significant]

    if len(d_tau_significant) < 2:
        return 0.0

    sign_changes = np.sum(np.diff(np.sign(d_tau_significant)) != 0)
    duration = hist['t'][mask][-1] - hist['t'][mask][0]
    return sign_changes / duration


# =============================================================================
# SIMULAZIONE CON PROFILO DI CORRENTE A STEP (per confronto con ROS2)
# =============================================================================
def run_simulation_steps(controller_type, x_ref, y_ref, psi_ref,
                          step_values, step_duration,
                          current_direction_deg=0.0,
                          dt=0.01, control_dt=0.02, gains=None,
                          x0=0.0, y0=0.0, psi0=0.0):
    """
    Come run_simulation, ma la corrente non e' costante per tutto il run:
    resta ferma a step_values[i] per step_duration secondi, poi passa al
    valore successivo -- stessa identica logica di _current_magnitude()
    in marine_current_node (modalita' enable_steps), cosi' il confronto
    ROS2 vs offline usa lo stesso identico profilo.

    Ritorna lo stesso dizionario di history di run_simulation, con in
    piu' la chiave 'current_mag' (magnitudine di corrente effettiva ad
    ogni istante, utile per l'asse x dei grafici di confronto).
    """
    direction_rad = np.deg2rad(current_direction_deg)
    n_steps_seq = len(step_values)
    duration = n_steps_seq * step_duration

    if controller_type == 'pid':
        ctrl_u, ctrl_v, ctrl_r = make_pid_controllers(control_dt, gains)
    elif controller_type == 'smc':
        ctrl_u, ctrl_v, ctrl_r = make_smc_controllers(gains)
    else:
        raise ValueError("controller_type deve essere 'pid' o 'smc'")

    state = np.array([x0, y0, psi0, 0.0, 0.0, 0.0])   # x, y, psi, u, v, r
    n_iter = int(duration / dt)
    control_every = max(1, int(control_dt / dt))

    history = dict(t=[], x=[], y=[], psi=[], u=[], v=[], r=[],
                    tau_u=[], tau_v=[], tau_r=[], current_mag=[])
    tau = np.zeros(3)

    for i in range(n_iter):
        t = i * dt

        # Stessa logica di marine_current_node in modalita' step:
        # indice del gradino corrente, bloccato sull'ultimo a fine sequenza
        idx = min(int(t // step_duration), n_steps_seq - 1)
        mag = step_values[idx]
        current = np.array([mag * np.cos(direction_rad), mag * np.sin(direction_rad)])

        if i % control_every == 0:
            x, y, psi, u, v, r = state
            e_x = x_ref - x
            e_y = y_ref - y
            e_psi = wrap_angle(psi_ref - psi)

            if controller_type == 'pid':
                tau_u = ctrl_u.update(e_x)
                tau_v = ctrl_v.update(e_y)
                tau_r = ctrl_r.update(e_psi)
            else:
                tau_u = ctrl_u.update(e_x, u)
                tau_v = ctrl_v.update(e_y, v)
                tau_r = ctrl_r.update(e_psi, r)

            tau = np.array([tau_u, tau_v, tau_r])

        state = rk4_step(state, tau, current, dt)

        history['t'].append(t)
        history['x'].append(state[0])
        history['y'].append(state[1])
        history['psi'].append(state[2])
        history['u'].append(state[3])
        history['v'].append(state[4])
        history['r'].append(state[5])
        history['tau_u'].append(tau[0])
        history['tau_v'].append(tau[1])
        history['tau_r'].append(tau[2])
        history['current_mag'].append(mag)

    return {k: np.array(v) for k, v in history.items()}


def step_end_values(hist, step_duration, n_steps_seq, key='x',
                     avg_last_seconds=5.0, dt=0.01):
    """
    Estrae, per ogni gradino, la magnitudine di corrente e il valore di
    'key' mediato sugli ultimi avg_last_seconds del gradino (cioe' il
    valore a regime raggiunto in quel segmento, non l'ultimo campione
    singolo che potrebbe essere rumoroso).
    """
    n_avg = int(avg_last_seconds / dt)
    mags, values = [], []
    for idx in range(n_steps_seq):
        start = int(idx * step_duration / dt)
        end = int((idx + 1) * step_duration / dt)
        end = min(end, len(hist[key]))
        segment_end = hist[key][max(start, end - n_avg):end]
        mags.append(hist['current_mag'][end - 1])
        values.append(np.mean(segment_end))
    return np.array(mags), np.array(values)