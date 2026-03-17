import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import meshcat
from meshcat.animation import Animation

import birotor_with_payload_visualizer as vis
import birotor_with_payload_dynamics as dyn

## Equilibrium state
x0 = np.zeros(8)
u0 = g * (dyn.mass_P + dyn.mass_Q) / 2 * np.ones(2)

# Simulation
T = 1

sol = solve_ivp(
    lambda t, x: dyn.f(t, x, u0),
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
