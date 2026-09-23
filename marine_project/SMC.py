#!/usr/bin/env python3
"""
smc_controller_node.py

Controllore Sliding Mode 3-DOF (surge, sway, yaw) per il
net-distance-keeping del BlueROV2.

DERIVAZIONE (stessa logica per surge/sway/yaw)

    e = riferimento - valore attuale        (es. e_x = x_ref - x)
    e_dot ≈ -vel                            (vel = u, v o r)

    Superficie di scorrimento:
        s = e_dot + lambda*e  =  -vel + lambda*e

    Vogliamo che s converga a zero in modo controllato:
        s_dot = -k*sat(s/phi)

    Sostituendo la dinamica del ROV (I*vel_dot = tau + damping) e
    risolvendo per tau si ottiene:

        tau = -D_lin*vel - D_quad*|vel|*vel - I*lambda*vel + I*k*sat(s/phi)
              \____________ tau_eq ____________/   \_ termine switching _/

    tau_eq  -> conosce il modello, cancella l'attrito reale e spinge
               verso la convergenza voluta
    switch  -> corregge sempre verso s=0, qualunque disturbo (corrente)
               spinga il ROV fuori rotta, SENZA saperne il valore

    Stessa formula per i 3 assi perche' e_x, e_y, e_psi sono definiti
    tutti allo stesso modo -> una sola classe (AxisSMC) riusata 3 volte.

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


def sat(x, limit=1.0):
    return float(np.clip(x, -limit, limit))


class AxisSMC:

    def __init__(self, I, d_lin, d_quad, lam, k_switch, phi, tau_max):
        self.I = I
        self.d_lin = d_lin
        self.d_quad = d_quad
        self.lam = lam
        self.k_switch = k_switch
        self.phi = phi
        self.tau_max = tau_max

    def update(self, e, vel):
        s = -vel + self.lam * e
        tau_eq = -self.d_lin * vel - self.d_quad * abs(vel) * vel - self.I * self.lam * vel
        tau_switch = self.I * self.k_switch * sat(s / self.phi)
        tau = tau_eq + tau_switch
        return float(np.clip(tau, -self.tau_max, self.tau_max))


class SMCControllerNode(Node):

    def __init__(self):
        super().__init__('smc_controller_node')

        # --- Riferimenti del task -------------------------------------------------
        self.declare_parameter('net_x', 2.0)
        self.declare_parameter('d_ref', 1.1)
        self.declare_parameter('y_ref', 0.0)
        self.declare_parameter('psi_ref', 0.0)

        # --- Parametri idrodinamici nominali (DEVONO combaciare col dynamics_node) -
        self.declare_parameter('m_rb', 11.5)
        self.declare_parameter('m_a_u', 5.50)
        self.declare_parameter('m_a_v', 12.70)
        self.declare_parameter('iz_rigid', 0.16)
        self.declare_parameter('m_a_r', 0.12)
        self.declare_parameter('x_u', -4.03)
        self.declare_parameter('x_uu', -18.18)
        self.declare_parameter('y_v', -6.22)
        self.declare_parameter('y_vv', -21.66)
        self.declare_parameter('n_r', -0.07)
        self.declare_parameter('n_rr', -1.55)

        # --- Guadagni SMC (ts=4/lambda; k e phi partono prudenti, poi si tarano) ---
        self.declare_parameter('lambda_u', 0.5)   # ts ~ 8s
        self.declare_parameter('lambda_v', 0.5)   # ts ~ 8s
        self.declare_parameter('lambda_r', 1.0)   # ts ~ 4s
        self.declare_parameter('k_u', 3.55)
        self.declare_parameter('k_v', 4.70)
        self.declare_parameter('k_r', 2.0)
        self.declare_parameter('phi_u', 0.083)
        self.declare_parameter('phi_v', 0.09)
        self.declare_parameter('phi_r', 0.1)

        self.declare_parameter('tau_max_linear', 60.0)
        self.declare_parameter('tau_max_yaw', 5.0)
        self.declare_parameter('control_rate', 50.0)

        gp = self.get_parameter
        self.x_ref = gp('net_x').value - gp('d_ref').value
        self.y_ref = gp('y_ref').value
        self.psi_ref = gp('psi_ref').value

        I_u = gp('m_rb').value + gp('m_a_u').value
        I_v = gp('m_rb').value + gp('m_a_v').value
        I_z = gp('iz_rigid').value + gp('m_a_r').value

        self.smc_u = AxisSMC(I_u, gp('x_u').value, gp('x_uu').value,
                              gp('lambda_u').value, gp('k_u').value, gp('phi_u').value,
                              gp('tau_max_linear').value)
        self.smc_v = AxisSMC(I_v, gp('y_v').value, gp('y_vv').value,
                              gp('lambda_v').value, gp('k_v').value, gp('phi_v').value,
                              gp('tau_max_linear').value)
        self.smc_r = AxisSMC(I_z, gp('n_r').value, gp('n_rr').value,
                              gp('lambda_r').value, gp('k_r').value, gp('phi_r').value,
                              gp('tau_max_yaw').value)

        self.dt = 1.0 / gp('control_rate').value
        self.have_odom = False
        self.x = self.y = self.psi = 0.0
        self.u = self.v = self.r = 0.0

        self.create_subscription(Odometry, '/bluerov2/odom', self._odom_callback, 10)
        self.tau_pub = self.create_publisher(Wrench, '/bluerov2/tau_cmd', 10)
        self.timer = self.create_timer(self.dt, self._control_step)

        self.get_logger().info(
            f'SMC 3-DOF avviato: x_ref={self.x_ref:.2f}, y_ref={self.y_ref}, '
            f'psi_ref={self.psi_ref}, I_u={I_u:.2f}, I_v={I_v:.2f}, I_z={I_z:.3f}'
        )

    def _odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.psi = 2.0 * np.arctan2(qz, qw)

        self.u = msg.twist.twist.linear.x
        self.v = msg.twist.twist.linear.y
        self.r = msg.twist.twist.angular.z
        self.have_odom = True

    def _control_step(self):
        if not self.have_odom:
            return

        e_x = self.x_ref - self.x
        e_y = self.y_ref - self.y
        e_psi = wrap_angle(self.psi_ref - self.psi)

        tau_u = self.smc_u.update(e_x, self.u)
        tau_v = self.smc_v.update(e_y, self.v)
        tau_r = self.smc_r.update(e_psi, self.r)

        msg = Wrench()
        msg.force.x = tau_u
        msg.force.y = tau_v
        msg.torque.z = tau_r
        self.tau_pub.publish(msg)


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