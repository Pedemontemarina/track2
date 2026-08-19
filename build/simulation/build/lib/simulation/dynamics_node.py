#!/usr/bin/env python3
"""
Simula la dinamica 3-DOF (surge, sway, yaw) del BlueROV2, secondo il
modello ridotto di Fossen descritto nel capitolo 2 del paper
(capitolo2_modellazione.tex), equazioni (11)-(13).

STATO SIMULATO
    eta = [x, y, psi]   -> posizione e prua nel frame inerziale (NED, piano)
    nu  = [u, v, r]      -> velocita' surge, sway, yaw nel frame body

COSA FA IL NODO
    1. Riceve il comando di forza/momento (tau_u, tau_v, tau_r) dal nodo di
       controllo (PID o SMC), sul topic /bluerov2/tau_cmd.
    2. Riceve la velocita' di corrente marina (espressa nel frame inerziale),
       sul topic /bluerov2/current_velocity. Di default e' zero (acqua calma).
    3. Ad ogni passo di simulazione (100 Hz):
         - ruota la corrente nel frame body usando l'angolo di imbardata psi
         - calcola la velocita' relativa nu_r = nu - nu_c (eq. 8-9 del paper)
         - integra le equazioni di moto (eq. 11-13) con Runge-Kutta 4
    4. Pubblica lo stato aggiornato come nav_msgs/Odometry sul topic
       /bluerov2/odom (pose = eta, twist = nu), cosi' i nodi di controllo
       possono leggere posizione e velocita' del veicolo.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Vector3
from nav_msgs.msg import Odometry

from .core.dynamics import BlueROV2Dynamics


class BlueROV2DynamicsNode(Node):

    def __init__(self):
        super().__init__('bluerov2_dynamics_node')

        # ---------------------------------------------------------------
        # PARAMETRI FISICI (Tabella I del paper)
        # ---------------------------------------------------------------
        self.declare_parameter('m_rb', 11.5)           # massa rigida [kg]
        self.declare_parameter('iz', 0.16)             # inerzia yaw rigida [kg m^2]
        self.declare_parameter('m_a_u', -5.50)         # massa aggiunta surge [kg]
        self.declare_parameter('m_a_v', -12.70)        # massa aggiunta sway [kg]
        self.declare_parameter('m_a_r', -0.12)         # massa aggiunta yaw [kg m^2/rad]

        self.declare_parameter('x_u', -4.03)           # damping lineare surge
        self.declare_parameter('y_v', -6.22)           # damping lineare sway
        self.declare_parameter('n_r', -0.07)           # damping lineare yaw
        self.declare_parameter('x_uu', -18.18)         # damping quadratico surge
        self.declare_parameter('y_vv', -21.66)         # damping quadratico sway
        self.declare_parameter('n_rr', -1.55)          # damping quadratico yaw

        self.declare_parameter('dt', 0.01)             # passo di integrazione [s]
        self.declare_parameter('x0', 0.0)              # stato iniziale
        self.declare_parameter('y0', 0.0)
        self.declare_parameter('psi0', 0.0)

        gp = self.get_parameter

        self.dyn = BlueROV2Dynamics(
            m_rb=gp('m_rb').value, iz=gp('iz').value,
            m_a_u=gp('m_a_u').value, m_a_v=gp('m_a_v').value, m_a_r=gp('m_a_r').value,
            x_u=gp('x_u').value, y_v=gp('y_v').value, n_r=gp('n_r').value,
            x_uu=gp('x_uu').value, y_vv=gp('y_vv').value, n_rr=gp('n_rr').value,
        )
        self.dt = gp('dt').value

        self.get_logger().info(
            f'Inerzie effettive -> I_u={self.dyn.I_u:.3f}, I_v={self.dyn.I_v:.3f}, '
            f'I_z={self.dyn.I_z:.3f} kg (m^2)'
        )

        # ---------------------------------------------------------------
        # STATO: [x, y, psi, u, v, r]
        # ---------------------------------------------------------------
        self.state = np.array([
            gp('x0').value, gp('y0').value, gp('psi0').value,
            0.0, 0.0, 0.0
        ], dtype=float)

        # Ultimo comando ricevuto dal controllore: [tau_u, tau_v, tau_r]
        self.tau = np.zeros(3)

        # Corrente marina nel frame inerziale: [u_c, v_c]. Zero = acqua calma.
        self.current_inertial = np.zeros(2)

        # ---------------------------------------------------------------
        # TOPIC
        # ---------------------------------------------------------------
        self.create_subscription(
            Wrench, '/bluerov2/tau_cmd', self._tau_callback, 10)
        self.create_subscription(
            Vector3, '/bluerov2/current_velocity', self._current_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, '/bluerov2/odom', 10)

        # Timer di simulazione: un passo di integrazione ogni dt secondi
        self.timer = self.create_timer(self.dt, self._step)

    # ---------------------------------------------------------------
    # CALLBACK
    # ---------------------------------------------------------------
    def _tau_callback(self, msg: Wrench):
        """Riceve il comando dal nodo di controllo (PID o SMC)."""
        self.tau = np.array([msg.force.x, msg.force.y, msg.torque.z])

    def _current_callback(self, msg: Vector3):
        """Riceve la velocita' di corrente marina (frame inerziale, m/s)."""
        self.current_inertial = np.array([msg.x, msg.y])

    # ---------------------------------------------------------------
    # PASSO DI SIMULAZIONE
    # ---------------------------------------------------------------
    def _step(self):
        """Un passo di integrazione RK4 (core.dynamics) + pubblicazione."""
        self.state = self.dyn.step(self.state, self.tau, self.current_inertial, self.dt)
        self._publish_odom()

    def _publish_odom(self):
        x, y, psi, u, v, r = self.state

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'bluerov2'

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        # Quaternione da yaw puro (roll = pitch = 0, coerente con le
        # ipotesi di stabilita' passiva del capitolo 2)
        msg.pose.pose.orientation.z = float(np.sin(psi / 2.0))
        msg.pose.pose.orientation.w = float(np.cos(psi / 2.0))

        # Twist espresso in body frame, come da convenzione ROS REP-103
        msg.twist.twist.linear.x = float(u)
        msg.twist.twist.linear.y = float(v)
        msg.twist.twist.angular.z = float(r)

        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BlueROV2DynamicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
