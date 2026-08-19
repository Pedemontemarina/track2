"""
Controllori SISO per-asse (surge/x, sway/y, yaw/psi), usati sia dai nodi
ROS2 (PID.py, SMC.py) sia dagli esperimenti headless.

Convenzione comune a entrambi i controllori:
    e(t)     = ref - pos          (errore di posizione/distanza)
    e_dot(t) = -rate              (derivata esatta, rate = u, v o r)
La distanza dalla rete e' d(t) = x(t) (rete nel piano x=0, ROV che si
avvicina diminuendo x), quindi per l'asse surge ref = d_ref e pos = x.
Per sway e yaw si tratta di loop di station-keeping (ref = 0).
"""

import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PIDAxis:
    """PID standard: tau = Kp e + Ki INT(e) + Kd e_dot (eq. Control Methods)."""

    def __init__(self, kp, ki, kd, integral_limit=None, angular=False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.angular = angular
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, ref, pos, rate, dt):
        e = wrap_to_pi(ref - pos) if self.angular else (ref - pos)
        e_dot = -rate

        self.integral += e * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        return self.kp * e + self.ki * self.integral + self.kd * e_dot


class SMCAxis:
    """
    Sliding Mode Control per un asse scalare (eq. Control Methods):
        s = e_dot + lambda * e
        tau = tau_eq - k * sat(s / phi)
    tau_eq e' ottenuto imponendo s_dot = 0 sul modello nominale (senza
    conoscenza della corrente: e' esattamente il disturbo non modellato
    che il termine switching deve rigettare).
    """

    def __init__(self, inertia, lin_damp, quad_damp, lam, k, phi, angular=False):
        self.I = inertia
        self.lin_damp = lin_damp
        self.quad_damp = quad_damp
        self.lam = lam
        self.k = k
        self.phi = phi
        self.angular = angular

    def reset(self):
        pass

    def compute(self, ref, pos, rate, dt):
        e = wrap_to_pi(ref - pos) if self.angular else (ref - pos)
        e_dot = -rate

        s = e_dot + self.lam * e

        # Controllo equivalente: cancella la dinamica nominale (drag) e
        # impone e_ddot = -lambda * e_dot (convergenza della sliding surface).
        tau_eq = -self.lin_damp * rate - self.quad_damp * abs(rate) * rate \
            - self.I * self.lam * rate

        # Nota di segno: con la convenzione e = ref - pos (coerente con il
        # PID), la legge stabilizzante e' tau = tau_eq + k*sat(s/phi), non
        # tau_eq - k*sat(s/phi). Verificato via Lyapunov (V=s^2/2, V_dot =
        # s*s_dot = -s*tau_sw/I, che e' <=0 solo se tau_sw ha lo stesso
        # segno di s, cioe' con il "+").
        sat = np.clip(s / self.phi, -1.0, 1.0)
        return tau_eq + self.k * sat
