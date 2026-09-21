#!/usr/bin/env python3
"""
Lancia insieme: dynamics_node + marine_current_node + UN controllore
(pid oppure smc, scelto con l'argomento 'controller').

    ros2 launch marine_project launch.py \
        controller:=pid \
        enable_steps:=true \
        step_values:="[0.3,0.7,0.85,0.9,1.0,1.2]" \
        step_duration:=70.0 \
        current_direction_deg:=0.0

    ros2 launch marine_project launch.py \
        controller:=smc \
        enable_steps:=true \
        step_values:="[0.3,0.7,0.85,0.9,1.0,1.2]" \
        step_duration:=70.0 \
        current_direction_deg:=0.0

In un altro terminale (prima o subito dopo aver lanciato):

    ros2 bag record /bluerov2/odom /bluerov2/tau_cmd -o bag_pid_steps

Durata totale della sequenza = n_step_values * step_duration
(nell'esempio sopra: 6 * 70 = 420 s).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node

PACKAGE_NAME = 'marine_project'


def generate_launch_description():

    controller_arg = DeclareLaunchArgument(
        'controller', default_value='pid',
        description="Controller: 'pid' or 'smc'"
    )

    # --- corrente: costante / rampa / step (vedi marine_current_node) ---------
    current_magnitude_arg = DeclareLaunchArgument('current_magnitude', default_value='0.0')
    current_direction_arg = DeclareLaunchArgument('current_direction_deg', default_value='0.0')

    enable_ramp_arg = DeclareLaunchArgument('enable_ramp', default_value='false')
    ramp_start_arg = DeclareLaunchArgument('ramp_start', default_value='0.0')
    ramp_end_arg = DeclareLaunchArgument('ramp_end', default_value='0.3')
    ramp_duration_arg = DeclareLaunchArgument('ramp_duration', default_value='60.0')

    enable_steps_arg = DeclareLaunchArgument('enable_steps', default_value='false')
    step_values_arg = DeclareLaunchArgument(
        'step_values', default_value='[0.3,0.7,0.85,0.9,1.0,1.2]')
    step_duration_arg = DeclareLaunchArgument('step_duration', default_value='70.0')

    # --- riferimenti del task (devono combaciare con offline_sim.py) ----------
    net_x_arg = DeclareLaunchArgument('net_x', default_value='2.0')
    d_ref_arg = DeclareLaunchArgument('d_ref', default_value='1.1')
    y_ref_arg = DeclareLaunchArgument('y_ref', default_value='0.0')
    psi_ref_arg = DeclareLaunchArgument('psi_ref', default_value='0.0')

    controller = LaunchConfiguration('controller')

    dynamics_node = Node(
        package=PACKAGE_NAME,
        executable='dynamics_node',
        name='bluerov2_dynamics_node',
        output='screen',
    )

    current_node = Node(
        package=PACKAGE_NAME,
        executable='currents',
        name='marine_current_node',
        output='screen',
        parameters=[{
            'current_magnitude': LaunchConfiguration('current_magnitude'),
            'current_direction_deg': LaunchConfiguration('current_direction_deg'),
            'enable_ramp': LaunchConfiguration('enable_ramp'),
            'ramp_start': LaunchConfiguration('ramp_start'),
            'ramp_end': LaunchConfiguration('ramp_end'),
            'ramp_duration': LaunchConfiguration('ramp_duration'),
            'enable_steps': LaunchConfiguration('enable_steps'),
            'step_values': LaunchConfiguration('step_values'),
            'step_duration': LaunchConfiguration('step_duration'),
        }],
    )

    pid_node = Node(
        package=PACKAGE_NAME,
        executable='PID',
        name='pid_controller_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", controller, "' == 'pid'"])),
        parameters=[{
            'net_x': LaunchConfiguration('net_x'),
            'd_ref': LaunchConfiguration('d_ref'),
            'y_ref': LaunchConfiguration('y_ref'),
            'psi_ref': LaunchConfiguration('psi_ref'),
        }],
    )

    smc_node = Node(
        package=PACKAGE_NAME,
        executable='SMC',
        name='smc_controller_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", controller, "' == 'smc'"])),
        parameters=[{
            'net_x': LaunchConfiguration('net_x'),
            'd_ref': LaunchConfiguration('d_ref'),
            'y_ref': LaunchConfiguration('y_ref'),
            'psi_ref': LaunchConfiguration('psi_ref'),
        }],
    )

    return LaunchDescription([
        controller_arg,
        current_magnitude_arg, current_direction_arg,
        enable_ramp_arg, ramp_start_arg, ramp_end_arg, ramp_duration_arg,
        enable_steps_arg, step_values_arg, step_duration_arg,
        net_x_arg, d_ref_arg, y_ref_arg, psi_ref_arg,
        dynamics_node,
        current_node,
        pid_node,
        smc_node,
    ])