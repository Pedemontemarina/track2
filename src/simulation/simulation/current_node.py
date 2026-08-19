#!/usr/bin/env python3
"""
Nodo ROS2 che pubblica il disturbo di corrente marina su
/bluerov2/current_velocity, per la demo live (la campagna di robustezza
vera gira headless dal notebook tramite core.experiment).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

from .core.currents import CurrentProfile


class CurrentDisturbanceNode(Node):

    def __init__(self):
        super().__init__('bluerov2_current_node')

        self.declare_parameter('amplitude', 0.0)      # m/s
        self.declare_parameter('direction', 0.0)       # rad, frame inerziale
        self.declare_parameter('profile', 'step')       # 'step' o 'ramp'
        self.declare_parameter('t_onset', 5.0)          # s
        self.declare_parameter('ramp_duration', 5.0)    # s
        self.declare_parameter('publish_rate', 20.0)    # Hz

        gp = self.get_parameter
        self.current = CurrentProfile(
            amplitude=gp('amplitude').value, direction=gp('direction').value,
            profile=gp('profile').value, t_onset=gp('t_onset').value,
            ramp_duration=gp('ramp_duration').value,
        )

        self.pub = self.create_publisher(Vector3, '/bluerov2/current_velocity', 10)
        self.t0 = self.get_clock().now()
        rate = gp('publish_rate').value
        self.create_timer(1.0 / rate, self._publish)

    def _publish(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        uc, vc = self.current(t)
        self.pub.publish(Vector3(x=uc, y=vc, z=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = CurrentDisturbanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
