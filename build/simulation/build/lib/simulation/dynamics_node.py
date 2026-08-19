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
        self.declare_parameter('combine_yaw_inertia', True)

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

        m_rb = gp('m_rb').value
        self.I_u = m_rb - gp('m_a_u').value
        self.I_v = m_rb - gp('m_a_v').value
        self.I_z = gp('iz').value - gp('m_a_r').value

        self.X_u = gp('x_u').value
        self.Y_v = gp('y_v').value
        self.N_r = gp('n_r').value
        self.X_uu = gp('x_uu').value
        self.Y_vv = gp('y_vv').value
        self.N_rr = gp('n_rr').value

        self.dt = gp('dt').value

        self.get_logger().info(
            f'Inerzie effettive -> I_u={self.I_u:.3f}, I_v={self.I_v:.3f}, '
            f'I_z={self.I_z:.3f} kg (m^2)'
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
    # MODELLO DINAMICO (eq. 8-13 del paper)
    # ---------------------------------------------------------------
    def _dynamics(self, state, tau, current_inertial):
        """Restituisce state_dot = f(state, tau, corrente)."""
        x, y, psi, u, v, r = state
        tau_u, tau_v, tau_r = tau

        cos_p, sin_p = np.cos(psi), np.sin(psi)

        # Corrente ruotata dal frame inerziale al frame body (eq. 8-9):
        # se il ROV e' allineato con la corrente, u_r si riduce alla
        # differenza scalare; altrimenti la rotazione la scompone
        # correttamente su surge e sway.
        uc_n, vc_n = current_inertial
        uc_b = cos_p * uc_n + sin_p * vc_n
        vc_b = -sin_p * uc_n + cos_p * vc_n

        u_r = u - uc_b
        v_r = v - vc_b

        # Equazioni di moto (eq. 11-13): I * nu_dot = tau - drag(nu_r)
        # Nota: lo yaw non e' influenzato dalla corrente (solo surge/sway,
        # vedi eq. 9 del paper), quindi qui uso r e non una velocita' relativa.
        u_dot = (tau_u + self.X_u * u_r + self.X_uu * abs(u_r) * u_r) / self.I_u
        v_dot = (tau_v + self.Y_v * v_r + self.Y_vv * abs(v_r) * v_r) / self.I_v
        r_dot = (tau_r + self.N_r * r + self.N_rr * abs(r) * r) / self.I_z

        # Cinematica planare: eta_dot = J(psi) * nu
        x_dot = u * cos_p - v * sin_p
        y_dot = u * sin_p + v * cos_p
        psi_dot = r

        return np.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot])

    def _step(self):
        """Un passo di integrazione RK4 + pubblicazione dello stato."""
        s = self.state
        tau = self.tau
        cur = self.current_inertial
        dt = self.dt

        k1 = self._dynamics(s, tau, cur)
        k2 = self._dynamics(s + dt / 2 * k1, tau, cur)
        k3 = self._dynamics(s + dt / 2 * k2, tau, cur)
        k4 = self._dynamics(s + dt * k3, tau, cur)

        self.state = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
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
