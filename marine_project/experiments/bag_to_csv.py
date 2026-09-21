#!/usr/bin/env python3
"""
bag_to_csv.py

Legge un bag ROS2 registrato con:
    ros2 bag record /bluerov2/odom /bluerov2/tau_cmd /bluerov2/current_velocity -o <nome_bag>

ed estrae timestamp, x, y, psi, u, v, r, tau_u, tau_v, tau_r,
current_x, current_y in un CSV, usando l'ultimo valore noto di
tau_cmd e current_velocity ad ogni messaggio di odom (i topic
viaggiano a frequenze diverse: 100Hz odom / 50Hz tau_cmd / 20Hz
current_velocity).

Il topic /bluerov2/current_velocity e' opzionale: se il bag non lo
contiene, le colonne current_x/current_y restano a 0.0 per tutte le
righe (compatibile con bag registrati senza quel topic).

USO
    python3 bag_to_csv.py <percorso_bag> <output.csv>

Richiede: rosbag2_py (viene con l'installazione standard di ROS2).
"""

import sys
import csv
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench, Vector3


def read_bag(bag_path, storage_id='mcap'):
    storage_options = StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    rows = []
    last_tau = (0.0, 0.0, 0.0)
    last_current = (0.0, 0.0)

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic == '/bluerov2/tau_cmd':
            msg = deserialize_message(data, Wrench)
            last_tau = (msg.force.x, msg.force.y, msg.torque.z)

        elif topic == '/bluerov2/current_velocity':
            msg = deserialize_message(data, Vector3)
            last_current = (msg.x, msg.y)

        elif topic == '/bluerov2/odom':
            msg = deserialize_message(data, Odometry)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            psi = 2.0 * np.arctan2(qz, qw)
            u = msg.twist.twist.linear.x
            v = msg.twist.twist.linear.y
            r = msg.twist.twist.angular.z

            t_s = t_ns * 1e-9
            rows.append([t_s, x, y, psi, u, v, r, *last_tau, *last_current])

    return rows


def main():
    if len(sys.argv) != 3:
        print('Uso: python3 bag_to_csv.py <percorso_bag> <output.csv>')
        sys.exit(1)

    bag_path, out_csv = sys.argv[1], sys.argv[2]
    rows = read_bag(bag_path)

    if not rows:
        print('Nessun dato trovato nel bag: controlla i nomi dei topic.')
        sys.exit(1)

    # Timestamp relativo (parte da zero) -- comodo per confrontare con
    # l'asse t di offline_sim.py, che parte anch'esso da 0
    t0 = rows[0][0]
    for row in rows:
        row[0] -= t0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'psi', 'u', 'v', 'r', 'tau_u', 'tau_v', 'tau_r',
                          'current_x', 'current_y'])
        writer.writerows(rows)

    print(f'Salvate {len(rows)} righe in {out_csv}')


if __name__ == '__main__':
    main()