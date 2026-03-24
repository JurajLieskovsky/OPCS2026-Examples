import numpy as np
import scipy
import matplotlib.pyplot as plt

import meshcat
from meshcat.animation import Animation

import birotor_with_payload_visualizer as vis
import birotor_with_payload_dynamics as dyn

## Equilibrium state
x_eq = np.zeros(8)
u_eq = dyn.g * (dyn.mass_P + dyn.mass_Q) / 2 * np.ones(2)

A, B = dyn.df(0, x_eq, u_eq)

## LQR design
Q = 1e2 * np.identity(8)
R = np.identity(2)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)

print(K)

# Simulation
T = 8

x0 = np.array([5, 2, 0, 0, 0, 0, 0, 0])

sol = scipy.integrate.solve_ivp(
    lambda t, x: dyn.f(t, x, u_eq - K @ (x - x_eq)),
    [0.0, T],
    x0,
    dense_output=True,
)


# Visualization
tspan = np.linspace(0, T, 100)

fix, ax = plt.subplots()
for i in range(4):
    ax.plot(tspan, sol.sol(tspan)[i, :], label=f"x{i}")

ax.legend()
plt.show(block=False)

#  animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)

anim = Animation(default_framerate=len(tspan) / T)
for i, t in enumerate(tspan):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, sol.sol(t))

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
