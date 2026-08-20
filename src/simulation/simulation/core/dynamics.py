"""
Modello 3-DOF (surge, sway, yaw) del BlueROV2.

Stato: [x, y, psi, u, v, r]
    x, y, psi -> posizione e prua nel frame inerziale (piano NED)
    u, v, r   -> velocita' surge, sway, yaw nel frame body

Calcolo il nuovo statp usando le equazioni di Fossen per il modello disaccoppiato e
integro numericamente usando RK4. Il metodo piu semplice sarebbe Eulero che mantiene una 
pendenza costante in tutto l'intervallo dt ma in questo modo accumulerei un errore. 
Con RK4 invece calcolo la pendenza in 4 punti diversi e faccio una media pesata, ottenendo 
un risultato piu' accurato.
-k1 = pendenza inizio intervallo
-k2 = pendenza a meta' intervallo usando k1 per stimare lo stato
-k3 = pendenza a meta' intervallo usando k2 per stimare lo stato
-k4 = pendenza a fine intervallo usando k3 per stimare lo stato

questo mi permette di usare un intervallo dt piu grande senza perdere precisione,
e quindi di avere una simulazione piu' veloce.
"""

import numpy as np


DEFAULT_PARAMS = dict(
    m_rb=11.5, iz=0.16,
    m_a_u=-5.50, m_a_v=-12.70, m_a_r=-0.12,
    x_u=-4.03, y_v=-6.22, n_r=-0.07,
    x_uu=-18.18, y_vv=-21.66, n_rr=-1.55,
)


class BlueROV2Dynamics:

    def __init__(self, **params):
        p = {**DEFAULT_PARAMS, **params}

        # Inerzia effettiva = massa rigida - massa aggiunta (m_a_* e' negativa
        # per convenzione di Fossen).
        self.I_u = p['m_rb'] - p['m_a_u']
        self.I_v = p['m_rb'] - p['m_a_v']
        self.I_z = p['iz'] - p['m_a_r']

        self.X_u, self.Y_v, self.N_r = p['x_u'], p['y_v'], p['n_r']
        self.X_uu, self.Y_vv, self.N_rr = p['x_uu'], p['y_vv'], p['n_rr']

    def derivative(self, state, tau, current_inertial):
        """state_dot = f(state, tau, corrente), eq. (8)-(13) del paper."""
        x, y, psi, u, v, r = state
        tau_u, tau_v, tau_r = tau

        cos_p, sin_p = np.cos(psi), np.sin(psi)

        """corrente nel frame del robot"""
        uc_n, vc_n = current_inertial
        uc_b = cos_p * uc_n + sin_p * vc_n
        vc_b = -sin_p * uc_n + cos_p * vc_n

        u_r = u - uc_b
        v_r = v - vc_b

        u_dot = (tau_u + self.X_u * u_r + self.X_uu * abs(u_r) * u_r) / self.I_u
        v_dot = (tau_v + self.Y_v * v_r + self.Y_vv * abs(v_r) * v_r) / self.I_v
        r_dot = (tau_r + self.N_r * r + self.N_rr * abs(r) * r) / self.I_z

        """riporto nel frame inerziale"""
        x_dot = u * cos_p - v * sin_p
        y_dot = u * sin_p + v * cos_p
        psi_dot = r

        return np.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot])

    """integrazione numerica """
    def step(self, state, tau, current_inertial, dt):
        k1 = self.derivative(state, tau, current_inertial)
        k2 = self.derivative(state + dt / 2 * k1, tau, current_inertial)
        k3 = self.derivative(state + dt / 2 * k2, tau, current_inertial)
        k4 = self.derivative(state + dt * k3, tau, current_inertial)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
