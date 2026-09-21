#!/usr/bin/env python3
"""
marine_current_node.py

Genera la corrente marina e la pubblica su /bluerov2/current_velocity,
nel frame INERZIALE (il dynamics_node la ruota lui nel body frame).

TRE MODALITA' (mutuamente esclusive, in ordine di priorita':
enable_steps > enable_ramp > costante)

1) COSTANTE (default)
    current_magnitude       -> intensita' della corrente [m/s]
    current_direction_deg   -> da che direzione arriva
                                  0   = lungo +x (mette alla prova il surge)
                                  90  = lungo +y (mette alla prova il sway)
                                  45  = diagonale (mette alla prova entrambi)

2) RAMPA (enable_ramp:=true)
    la corrente cresce linearmente da ramp_start a ramp_end in
    ramp_duration secondi, poi resta costante al valore finale.
    NOTA: sconsigliata per trovare la soglia di rottura vicino al
    breakpoint, perche' il settling time del controllore in quella
    zona (~50-60s) e' paragonabile alla velocita' di salita della
    rampa: l'errore osservato in un dato istante e' un misto tra
    "quanto e' forte la corrente ora" e "quanto il controllore ha
    fatto in tempo a recuperare finora", quindi non rappresenta un
    vero steady-state error. Usare la modalita' a step (sotto) per
    un confronto valido con lo sweep offline.

3) STEP SEQUENCE (enable_steps:=true) <-- usata per la validazione ROS2 vs offline
    la corrente resta costante a step_values[i] per step_duration
    secondi, poi salta al valore successivo. Ogni segmento e'
    equivalente a un run offline separato (tempo sufficiente per
    arrivare a regime), ma tutti i segmenti sono catturati in un
    unico bag continuo.
    Parametri:
        step_values     -> lista di magnitudini [m/s], es. "[0.3,0.7,0.85,0.9,1.0,1.2]"
        step_duration    -> quanti secondi resta fermo su ogni valore [s]

INTERFACCIA
    Pubblica: /bluerov2/current_velocity   (geometry_msgs/Vector3, m/s, frame inerziale)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


class MarineCurrentNode(Node):

    def __init__(self):
        super().__init__('marine_current_node')

        # --- Modalita' costante -----------------------------------------------
        self.declare_parameter('current_magnitude', 0.0)       # [m/s]
        self.declare_parameter('current_direction_deg', 0.0)   # 0=+x, 90=+y

        # --- Modalita' rampa (opzionale) ----------------------------------------
        self.declare_parameter('enable_ramp', False)
        self.declare_parameter('ramp_start', 0.0)
        self.declare_parameter('ramp_end', 0.3)
        self.declare_parameter('ramp_duration', 60.0)

        # --- Modalita' step sequence (opzionale) --------------------------------
        self.declare_parameter('enable_steps', False)
        self.declare_parameter('step_values', [0.3, 0.7, 0.85, 0.9, 1.0, 1.2])
        self.declare_parameter('step_duration', 70.0)   # [s] per ogni gradino

        self.declare_parameter('publish_rate', 20.0)

        gp = self.get_parameter
        self.magnitude = gp('current_magnitude').value
        self.direction_rad = np.deg2rad(gp('current_direction_deg').value)

        self.enable_ramp = gp('enable_ramp').value
        self.ramp_start = gp('ramp_start').value
        self.ramp_end = gp('ramp_end').value
        self.ramp_duration = gp('ramp_duration').value

        self.enable_steps = gp('enable_steps').value
        self.step_values = list(gp('step_values').value)
        self.step_duration = gp('step_duration').value

        self.dt = 1.0 / gp('publish_rate').value
        self.t0 = self.get_clock().now()

        self.pub = self.create_publisher(Vector3, '/bluerov2/current_velocity', 10)
        self.timer = self.create_timer(self.dt, self._publish_current)

        if self.enable_steps:
            mode = 'STEP SEQUENCE'
            total_duration = len(self.step_values) * self.step_duration
            extra = (f'values={self.step_values} m/s, '
                     f'step_duration={self.step_duration}s, '
                     f'total={total_duration}s')
        elif self.enable_ramp:
            mode = 'RAMPA'
            extra = f'{self.ramp_start} -> {self.ramp_end} m/s in {self.ramp_duration}s'
        else:
            mode = 'COSTANTE'
            extra = f'mag={self.magnitude} m/s'

        self.get_logger().info(
            f'Corrente avviata in modalita\' {mode} ({extra}), '
            f'dir={gp("current_direction_deg").value} deg'
        )

    def _current_magnitude(self):
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9

        if self.enable_steps:
            n_steps = len(self.step_values)
            idx = int(elapsed // self.step_duration)
            if idx >= n_steps:
                idx = n_steps - 1   # resta sull'ultimo valore a fine sequenza
            return self.step_values[idx]

        if self.enable_ramp:
            if elapsed >= self.ramp_duration:
                return self.ramp_end
            frac = elapsed / self.ramp_duration
            return self.ramp_start + frac * (self.ramp_end - self.ramp_start)

        return self.magnitude

    def _publish_current(self):
        mag = self._current_magnitude()

        msg = Vector3()
        msg.x = float(mag * np.cos(self.direction_rad))
        msg.y = float(mag * np.sin(self.direction_rad))
        msg.z = 0.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MarineCurrentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()