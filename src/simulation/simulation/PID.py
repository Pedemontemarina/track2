#!/usr/bin/env python3
"""
Nodo ROS2 di controllo PID per il task di net-distance-keeping.

Tre loop PID indipendenti :
    surge (x) -> mantiene la distanza dalla rete a d_ref
    sway  (y) -> station-keeping laterale a y_ref (default 0)
    yaw (psi) -> mantiene la prua a psi_ref (default 0, perpendicolare alla rete)

Legge, per ciascun asse (eq. Control Methods):
    tau(t) = Kp e(t) + Ki INT(e) + Kd e_dot(t),   e(t) = ref - pos(t)

Riceve nav_msgs/Odometry su /bluerov2/odom, pubblica geometry_msgs/Wrench
su /bluerov2/tau_cmd.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry

from .core.controllers import PIDAxis
from .core.experiment import DEFAULT_GAINS


def yaw_from_quaternion(q):
    return 2.0 * np.arctan2(q.z, q.w)


class PIDControllerNode(Node):

    def __init__(self):
        super().__init__('bluerov2_pid_node')

        g = DEFAULT_GAINS['PID']
        self.declare_parameter('d_ref', 1.10) # distanza dalla rete in metri
        self.declare_parameter('y_ref', 0.0)  # posizione laterale desiderata in metri (0 = centro canale)
        self.declare_parameter('psi_ref', 0.0)  # prua desiderata in radianti (0 = perpendicolare alla rete)
        self.declare_parameter('tau_limit', 30.0)  # limite della forza/torque
        self.declare_parameter('kp_surge', g['surge']['kp'])
        self.declare_parameter('ki_surge', g['surge']['ki'])
        self.declare_parameter('kd_surge', g['surge']['kd'])
        self.declare_parameter('kp_sway', g['sway']['kp'])
        self.declare_parameter('ki_sway', g['sway']['ki'])
        self.declare_parameter('kd_sway', g['sway']['kd'])
        self.declare_parameter('kp_yaw', g['yaw']['kp'])
        self.declare_parameter('ki_yaw', g['yaw']['ki'])
        self.declare_parameter('kd_yaw', g['yaw']['kd'])
        #parametri configurabili

        gp = self.get_parameter
        self.d_ref = gp('d_ref').value
        self.y_ref = gp('y_ref').value
        self.psi_ref = gp('psi_ref').value
        self.tau_limit = gp('tau_limit').value

        self.ctrl = dict(
            surge=PIDAxis(gp('kp_surge').value, gp('ki_surge').value, gp('kd_surge').value,
                           integral_limit=5.0),
            sway=PIDAxis(gp('kp_sway').value, gp('ki_sway').value, gp('kd_sway').value,
                          integral_limit=5.0),
            yaw=PIDAxis(gp('kp_yaw').value, gp('ki_yaw').value, gp('kd_yaw').value,
                        integral_limit=2.0, angular=True),
        )

        self.state = None  # (x, y, psi, u, v, r), None finche' non arriva odom
        self.last_t = None

        self.create_subscription(Odometry, '/bluerov2/odom', self._odom_callback, 10)
        self.tau_pub = self.create_publisher(Wrench, '/bluerov2/tau_cmd', 10)

    # ogni volta che arriva un messaggio di odometria, calcola l'errore e la forza/torque da applicare
    def _odom_callback(self, msg: Odometry):
        now = self.get_clock().now()
        # nel notebook simulo con un dt fisso a 0.01, ma qui calcolo il dt reale tra due callback di odometria
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
