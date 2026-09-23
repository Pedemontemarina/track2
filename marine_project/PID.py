#!/usr/bin/env python3
"""
pid_controller_node.py

Controllore PID a 3 gradi di liberta' (surge, sway, yaw) per il
net-distance-keeping del BlueROV2 sotto corrente marina proveniente
da direzione arbitraria.

CONVENZIONE DEGLI ERRORI (uniforme sui 3 assi, per evitare bug di segno)
    Tutti gli errori sono definiti come "riferimento - valore attuale":
        e_x   = x_ref   - x        (x_ref = net_x - d_ref, target di distanza)
        e_y   = y_ref   - y        (di default 0 = centerline rispetto alla rete)
        e_psi = wrap(psi_ref - psi) (di default 0 = ROV rivolto verso la rete)

# --- COME FUNZIONA x_ref ----------------------------------------------------
 La rete e' un piano fisso a coordinata net_x lungo l'asse x.
 La distanza dal ROV alla rete e': d = net_x - x
 Vogliamo d = d_ref (es. 1.1 m) -> risolvendo per x:
       x_ref = net_x - d_ref
 Cioe': x_ref e' la coordinata x in cui, se il ROV ci sta fermo, la sua
 distanza dalla rete e' esattamente d_ref. Da qui in poi il controllore
 non ragiona piu' in termini di "distanza", ma di "punto x da raggiungere",
 esattamente come fa gia' per y (y_ref) e psi (psi_ref).
# -----------------------------------------------------------------------------

    Nota su e_psi: l'errore di heading e' "wrappato" con atan2(sin,cos)
    per evitare il salto di +-pi quando l'angolo attraversa +-180 gradi.

    Ogni asse e' un loop PID indipendente (il modello e' disaccoppiato
    per costruzione).


INTERFACCIA
    Sottoscrive : /bluerov2/odom       (nav_msgs/Odometry)
    Pubblica    : /bluerov2/tau_cmd    (geometry_msgs/Wrench)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class AxisPID:
    """Un loop PID indipendente per un asse (surge, sway o yaw)."""

    def __init__(self, kp, ki, kd, tau_max, integral_max, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.tau_max = tau_max
        self.integral_max = integral_max
        self.dt = dt
        self.integral_e = 0.0
        self.prev_e = 0.0

    def update(self, e): # riceve l'errore e calcola la forza/momento da applicare
        self.integral_e += e * self.dt
        #anti windup
        self.integral_e = float(np.clip(self.integral_e, -self.integral_max, self.integral_max))
        # quanto veloce l'errore cambia
        e_dot = (e - self.prev_e) / self.dt
        self.prev_e = e

        tau = self.kp * e + self.ki * self.integral_e + self.kd * e_dot
        return float(np.clip(tau, -self.tau_max, self.tau_max))


class PIDControllerNode(Node):

    def __init__(self):
        super().__init__('pid_controller_node')

        # --- Riferimenti del task -------------------------------------------------
        self.declare_parameter('net_x', 2.0)
        self.declare_parameter('d_ref', 1.1)
        self.declare_parameter('y_ref', 0.0)
        self.declare_parameter('psi_ref', 0.0)

        # --- Guadagni SURGE (I_u=17, c_u=4.03 -> ts=8s, zeta=0.8) -----------------
        self.declare_parameter('kp_u', 7.5)
        self.declare_parameter('ki_u', 0.65)
        self.declare_parameter('kd_u', 11.0)

        # --- Guadagni SWAY (I_v=24.2, c_v=6.22 -> ts=8s, zeta=0.8) ----------------
        self.declare_parameter('kp_v', 9.5)
        self.declare_parameter('ki_v', 0.85)
        self.declare_parameter('kd_v', 20.0)

        # --- Guadagni YAW (I_z=0.28, c_r=0.07 -> ts=4s, zeta=0.8) -----------------
        self.declare_parameter('kp_r', 0.44)
        self.declare_parameter('ki_r', 0.055)
        self.declare_parameter('kd_r', 0.49)

        # --- Saturazioni e anti-windup ---------------------------------------------
        self.declare_parameter('tau_max_linear', 60.0)   # [N] surge/sway
        self.declare_parameter('tau_max_yaw', 5.0)        # [N.m]
        self.declare_parameter('integral_max_linear', 25.0)
        self.declare_parameter('integral_max_yaw', 1.0)

        self.declare_parameter('control_rate', 50.0)

        gp = self.get_parameter
        self.x_ref = gp('net_x').value - gp('d_ref').value
        self.y_ref = gp('y_ref').value
        self.psi_ref = gp('psi_ref').value

        dt = 1.0 / gp('control_rate').value
        self.dt = dt

        self.pid_u = AxisPID(gp('kp_u').value, gp('ki_u').value, gp('kd_u').value,
                              gp('tau_max_linear').value, gp('integral_max_linear').value, dt)
        self.pid_v = AxisPID(gp('kp_v').value, gp('ki_v').value, gp('kd_v').value,
                              gp('tau_max_linear').value, gp('integral_max_linear').value, dt)
        self.pid_r = AxisPID(gp('kp_r').value, gp('ki_r').value, gp('kd_r').value,
                              gp('tau_max_yaw').value, gp('integral_max_yaw').value, dt)

        self.have_odom = False
        self.x = self.y = self.psi = 0.0

        self.create_subscription(Odometry, '/bluerov2/odom', self._odom_callback, 10)
        self.tau_pub = self.create_publisher(Wrench, '/bluerov2/tau_cmd', 10)
        self.timer = self.create_timer(dt, self._control_step)

        self.get_logger().info(
            f'PID 3-DOF avviato: x_ref={self.x_ref:.2f} (d_ref={gp("d_ref").value}), '
            f'y_ref={self.y_ref}, psi_ref={self.psi_ref}'
        )

    def _odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.psi = 2.0 * np.arctan2(qz, qw)   # coerente con come il dynamics_node la pubblica
        self.have_odom = True

    def _control_step(self):
        if not self.have_odom:
            return

        e_x = self.x_ref - self.x
        e_y = self.y_ref - self.y
        e_psi = wrap_angle(self.psi_ref - self.psi)

        tau_u = self.pid_u.update(e_x)
        tau_v = self.pid_v.update(e_y)
        tau_r = self.pid_r.update(e_psi)

        msg = Wrench()
        msg.force.x = tau_u
        msg.force.y = tau_v
        msg.torque.z = tau_r
        self.tau_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()