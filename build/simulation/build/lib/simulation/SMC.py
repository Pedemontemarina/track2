#!/usr/bin/env python3
"""
Nodo ROS2 di controllo Sliding Mode (SMC) per il task di net-distance-keeping.

Tre superfici di sliding indipendenti, una per asse (surge/x, sway/y, yaw/psi),
speculari alla struttura del nodo PID (vedi PID.py):
    s(t)   = e_dot(t) + lambda * e(t),      e(t) = ref - pos(t)
    tau(t) = tau_eq - k * sat(s/phi)     -> vedi nota di segno in
                                            core/controllers.py: con questa
                                            convenzione di errore la legge
                                            stabilizzante e' in realta'
                                            tau_eq + k*sat(s/phi) (verificato
                                            via Lyapunov e in simulazione).

tau_eq compensa il modello nominale di damping (Tabella I) e il termine
inerziale I*lambda*rate, cosi' il solo switching term deve rigettare la
corrente non modellata: e' esattamente la proprieta' di robustezza che
questo lavoro vuole misurare.

Riceve nav_msgs/Odometry su /bluerov2/odom, pubblica geometry_msgs/Wrench
su /bluerov2/tau_cmd.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry

from .core.controllers import SMCAxis
from .core.dynamics import BlueROV2Dynamics
from .core.experiment import DEFAULT_GAINS


def yaw_from_quaternion(q):
    return 2.0 * np.arctan2(q.z, q.w)


class SMCControllerNode(Node):

    def __init__(self):
        super().__init__('bluerov2_smc_node')

        g = DEFAULT_GAINS['SMC']
        self.declare_parameter('d_ref', 1.10)
        self.declare_parameter('y_ref', 0.0)
        self.declare_parameter('psi_ref', 0.0)
        self.declare_parameter('tau_limit', 30.0)
        self.declare_parameter('lam_surge', g['surge']['lam'])
        self.declare_parameter('k_surge', g['surge']['k'])
        self.declare_parameter('phi_surge', g['surge']['phi'])
        self.declare_parameter('lam_sway', g['sway']['lam'])
        self.declare_parameter('k_sway', g['sway']['k'])
        self.declare_parameter('phi_sway', g['sway']['phi'])
        self.declare_parameter('lam_yaw', g['yaw']['lam'])
        self.declare_parameter('k_yaw', g['yaw']['k'])
        self.declare_parameter('phi_yaw', g['yaw']['phi'])

        gp = self.get_parameter
        self.d_ref = gp('d_ref').value
        self.y_ref = gp('y_ref').value
        self.psi_ref = gp('psi_ref').value
        self.tau_limit = gp('tau_limit').value

        # Parametri nominali del modello, usati dal controllo equivalente
        # (stessi valori di Tabella I: unica fonte di verita' condivisa
        # con dynamics_node tramite core.dynamics).
        dyn = BlueROV2Dynamics()

        self.ctrl = dict(
            surge=SMCAxis(dyn.I_u, dyn.X_u, dyn.X_uu,
                           gp('lam_surge').value, gp('k_surge').value, gp('phi_surge').value),
            sway=SMCAxis(dyn.I_v, dyn.Y_v, dyn.Y_vv,
                          gp('lam_sway').value, gp('k_sway').value, gp('phi_sway').value),
            yaw=SMCAxis(dyn.I_z, dyn.N_r, dyn.N_rr,
                        gp('lam_yaw').value, gp('k_yaw').value, gp('phi_yaw').value, angular=True),
        )

        self.last_t = None

        self.create_subscription(Odometry, '/bluerov2/odom', self._odom_callback, 10)
        self.tau_pub = self.create_publisher(Wrench, '/bluerov2/tau_cmd', 10)

    def _odom_callback(self, msg: Odometry):
        now = self.get_clock().now()
        dt = 0.01 if self.last_t is None else max((now - self.last_t).nanoseconds * 1e-9, 1e-4)
        self.last_t = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        psi = yaw_from_quaternion(msg.pose.pose.orientation)
        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        r = msg.twist.twist.angular.z

        tau_u = self.ctrl['surge'].compute(self.d_ref, x, u, dt)
        tau_v = self.ctrl['sway'].compute(self.y_ref, y, v, dt)
        tau_r = self.ctrl['yaw'].compute(self.psi_ref, psi, r, dt)
        tau = np.clip([tau_u, tau_v, tau_r], -self.tau_limit, self.tau_limit)

        out = Wrench()
        out.force.x, out.force.y, out.torque.z = (float(t) for t in tau)
        self.tau_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SMCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
