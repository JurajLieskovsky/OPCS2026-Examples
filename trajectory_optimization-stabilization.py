from cvxpy.atoms.quad_form import quad_form
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cvxpy as cp

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

## inf horizon LQR design
Q = 1e2 * np.identity(8)
R = np.identity(2)

Q_N = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ Q_N @ B, B.T @ Q_N @ A)

print(K)

## Trajectory optimization
N = 1000

x = cp.Variable((8, N + 1))
u = cp.Variable((2, N))
x_init = cp.Parameter(8)

constraints = [
    x[:, 1:] == A @ x[:, :-1] + B @ u,
    x[:, 0] == x_init,
    u >= -u_eq[:, np.newaxis]
]

LQ = np.linalg.cholesky(Q)
LR = np.linalg.cholesky(R)

objective = cp.Minimize(
    cp.sum_squares(LQ.T @ x[:, :-1])
    + cp.sum_squares(LR.T @ u)
    + quad_form(x[:, N], Q_N)
)

problem = cp.Problem(objective, constraints)

x0 = np.array([5, 2, 0, 0, 0, 0, 0, 0])

x_init.value = x0
problem.solve()

xs = [x_eq + x.value[:, k] for k in range(N + 1)]
us = [u_eq + u.value[:, k] for k in range(N)]

us.append(u_eq + u.value[:, N-1])

# Visualization
tspan = [h * k for k in range(N + 1)]

fix, ax = plt.subplots(2)
for i in range(4):
    ax[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")

for i in range(2):
    ax[1].step(tspan, [u[i] for u in us], where="post", label=f"u{i}")

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
