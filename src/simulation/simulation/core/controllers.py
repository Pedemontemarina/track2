"""
Controllori SISO per-asse (surge/x, sway/y, yaw/psi), usati sia dai nodi
ROS2 (PID.py, SMC.py) sia dagli esperimenti senza ROS2.

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

 """PID standard: tau = Kp e + Ki INT(e) + Kd e_dot (eq. Control Methods)."""
class PIDAxis:

    def __init__(self, kp, ki, kd, integral_limit=None, angular=False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit #limite max integrale
        self.angular = angular #flag per attivare wrap_to_pi per errori angolari
        self.integral = 0.0 

    def reset(self):
        self.integral = 0.0

    # calcola l'uscita del controllore PID per un asse scalare (surge, sway o yaw)
    # ref: posizione desiderata (distanza o angolo)
    # pos: posizione attuale (distanza o angolo)
    # rate: velocità attuale (u, v o r)
    # dt: intervallo di tempo tra i campioni
    def compute(self, ref, pos, rate, dt):
        e = wrap_to_pi(ref - pos) if self.angular else (ref - pos)
        e_dot = -rate

        # Aggiorna l'integrale dell'errore e applica il limite se specificato per evitare overshooting
        self.integral += e * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)


        return self.kp * e + self.ki * self.integral + self.kd * e_dot


class SMCAxis:
    """
    Sliding Mode Control per un asse scalare :
        s = e_dot + lambda * e
        tau = tau_eq - k * sat(s / phi)

    lambda > 0: regola la velocità di convergenza della sliding surface
    k > 0: guadagno del termine switching ossia forza della spinta correttiva (rigetta disturbi non modellati) 
    phi > 0: ampiezza della banda di saturazione (riduce chattering, ma aumenta l'errore di regime)
    
    """

    def __init__(self, inertia, lin_damp, quad_damp, lam, k, phi, angular=False):
        self.I = inertia # inerzia dell'asse
        self.lin_damp = lin_damp # coefficiente di dumping lineare 
        self.quad_damp = quad_damp # coefficiente di dumping quadratico 
        self.lam = lam # regola la velocità di convergenza della sliding surface
        self.k = k # guadagno del termine switching
        self.phi = phi # ampiezza della banda di saturazione
        self.angular = angular

    def reset(self):
        pass #non c'è memoria tra una chiamata e l'altra.

    def compute(self, ref, pos, rate, dt):
        e = wrap_to_pi(ref - pos) if self.angular else (ref - pos)
        e_dot = -rate

        s = e_dot + self.lam * e #superficie

        # Controllo equivalente, forza per restare su s=0: cancella la dinamica nominale (drag) e
        # impone e_ddot = -lambda * e_dot (convergenza della sliding surface).

        tau_eq = -self.lin_damp * rate - self.quad_damp * abs(rate) * rate \
            - self.I * self.lam * rate

        # switching term, forza correttiva per riportare s a 0.
        # sarebbe una funzione segno ma un discontinuità causa chattering (salti continui).
        # La funzione di saturazione è più 'morbida' e riduce il chattering.
        sat = np.clip(s / self.phi, -1.0, 1.0)
        return tau_eq + self.k * sat
