"""Profili di corrente marina per la campagna di robustezza."""

import numpy as np


class CurrentProfile:
    """
    Corrente nel frame inerziale, applicata a partire da t_onset:
        'step' -> gradino pieno ad ampiezza costante
        'ramp' -> rampa lineare da 0 ad ampiezza in ramp_duration secondi,
                  poi costante
    amplitude [m/s], direction [rad] (0 = lungo +x, verso il ROV che si
    allontana dalla rete se il ROV parte con x>0 avvicinandosi in -x).
    """

    def __init__(self, amplitude, direction=0.0, profile='step',
                 t_onset=5.0, ramp_duration=5.0):
        self.amplitude = amplitude
        self.direction = direction
        self.profile = profile
        self.t_onset = t_onset
        self.ramp_duration = ramp_duration

    def __call__(self, t):
        if t < self.t_onset:
            mag = 0.0
        elif self.profile == 'step':
            mag = self.amplitude
        elif self.profile == 'ramp':
            frac = min(1.0, (t - self.t_onset) / self.ramp_duration)
            mag = self.amplitude * frac
        else:
            raise ValueError(f'unknown profile {self.profile}')

        return np.array([mag * np.cos(self.direction), mag * np.sin(self.direction)])
