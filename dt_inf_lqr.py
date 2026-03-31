import numpy as np
import scipy
import matplotlib.pyplot as plt

import meshcat
from meshcat.animation import Animation

import birotor_with_payload_visualizer as vis
import birotor_with_payload_dynamics as dyn

## Timestep
h = 1e-2

## Equilibrium state
x_eq = np.zeros(8)
u_eq = dyn.g * (dyn.mass_P + dyn.mass_Q) / 2 * np.ones(2)

continuous_A, continuous_B = dyn.df(0, x_eq, u_eq)

A = np.identity(8) + h * continuous_A
B = h * continuous_B

## LQR design
Q = 1e2 * np.identity(8)
R = np.identity(2)

P = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

print(K)

# Simulation
N = 500
x0 = np.array([5, 2, 0, 0, 0, 0, 0, 0])

solver = scipy.integrate.ode(dyn.f)
solver.set_integrator(name="dopri5")
solver.set_initial_value(x0)

xs = [np.zeros(8) for _ in range(N+1)]
us = [np.zeros(2) for _ in range(N+1)]

xs[0] = solver.y

for k in range(N):
    us[k] = u_eq - K @ (xs[k] - x_eq)
    solver.set_f_params(us[k], np.random.normal(0.0, 1e0, (1,)))
    solver.integrate(solver.t+h)
    xs[k+1] = solver.y

us[N] = us[N-1]

# Visualization
tspan = [h * k for k in range(N+1)]

fix, ax = plt.subplots(2)
for i in range(4):
    ax[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")

for i in range(2):
    ax[1].step(tspan, [u[i] for u in us], where='post', label=f"u{i}")

ax[0].legend()
ax[1].legend()
plt.show(block=False)

#  animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)
vis.set_birotor_state(visualizer, x0)

anim = Animation(default_framerate=1 / h)
for i, x in enumerate(xs):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, x)

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
