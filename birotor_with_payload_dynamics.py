import numpy as np

# Model of the system
## Parameters
g = 9.81
mass_Q = 0.5  # kg
moi_Q = 4e-3  # kg * m^2
a = 0.175  # m

mass_P = 0.1  # kg
l = 0.5  # m


## Dynamics
def f(t, x, u):
    (y, z, theta, phi, dy, dz, dtheta, dphi) = x

    M = np.array(
        [
            [mass_Q + mass_P, 0, 0, l * mass_P * np.cos(phi)],
            [0, mass_Q + mass_P, 0, l * mass_P * np.sin(phi)],
            [0, 0, moi_Q, 0],
            [l * mass_P * np.cos(phi), l * mass_P * np.sin(phi), 0, l**2 * mass_P],
        ]
    )

    c = np.array(
        [
            -l * mass_P * np.sin(phi) * dphi**2,
            l * mass_P * np.cos(phi) * dphi**2,
            0,
            0,
        ]
    )

    tau_g = np.array(
        [
            0,
            -g * (mass_P + mass_Q),
            0,
            -g * l * mass_P * np.sin(phi),
        ]
    )

    B = np.array(
        [
            [-np.sin(theta), -np.sin(theta)],
            [np.cos(theta), np.cos(theta)],
            [-a, a],
            [0, 0],
        ]
    )

    return np.concatenate(
        (
            x[4:],
            np.linalg.solve(M, -c + tau_g + B @ u),
        )
    )
