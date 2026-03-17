import numpy as np
from meshcat.geometry import Box, Cylinder, Sphere, MeshBasicMaterial
import meshcat.transformations as tf


def set_birotor(vis, width, height, radius, length):
    blue = MeshBasicMaterial(color=0x0000FF)
    green = MeshBasicMaterial(color=0x00C000)

    # Body
    vis["quadrotor"]["body"].set_object(
        Box([width, width, height]),
        green,
    )

    # Pendulum
    vis["quadrotor"]["payload"]["pole"].set_object(
        Cylinder(length, width / 25),
        blue,
    )
    vis["quadrotor"]["payload"]["pole"].set_transform(
        tf.translation_matrix([0, 0, -length / 2])
        @ tf.quaternion_matrix([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    )

    vis["quadrotor"]["payload"]["mass"].set_object(
        Sphere(width / 6),
        green,
    )
    vis["quadrotor"]["payload"]["mass"].set_transform(
        tf.translation_matrix([0, 0, -length])
    )

    # Propellers
    positions = [
        [width / 2, -width / 2, 3 / 4 * height],
        [width / 2, width / 2, 3 / 4 * height],
        [-width / 2, width / 2, 3 / 4 * height],
        [-width / 2, -width / 2, 3 / 4 * height],
    ]

    for i, pos in enumerate(positions):
        vis["quadrotor"]["rotor" + str(i)].set_object(
            Cylinder(height / 2, radius), blue
        )
        vis["quadrotor"]["rotor" + str(i)].set_transform(
            tf.translation_matrix(pos)
            @ tf.quaternion_matrix([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        )

    return vis


def set_birotor_state(vis, state):
    vis["quadrotor"].set_transform(
        tf.translation_matrix([0, state[0], state[1]])
        @ tf.quaternion_matrix([np.cos(state[2] / 2), np.sin(state[2] / 2), 0, 0])
    )
    vis["quadrotor"]["payload"].set_transform(
        tf.quaternion_matrix([np.cos((state[3] - state[2]) / 2), np.sin((state[3] - state[2]) / 2), 0, 0])
    )
