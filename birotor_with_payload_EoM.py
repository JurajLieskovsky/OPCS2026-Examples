from sympy import symbols, Function, Matrix, diff, cos, sin, latex, simplify
from lagrangian2eom import massMatrix, velocityTerms, potentialTerms, wrenchMatrix

t = symbols("t")

y = Function("y")(t)
z = Function("z")(t)
theta = Function("theta")(t)
phi = Function("phi")(t)

g, mass_Q, moi_Q, mass_P, l, a = symbols("g m_Q I_Q m_P l a")

# Generalized coordinates
q = Matrix([y, z, theta, phi])

# Kinetic energy
vyP = diff(y, t) + l * diff(phi, t) * cos(phi)
vzP = diff(z, t) + l * diff(phi, t) * sin(phi)

T = 0.5 * (
    mass_Q * (diff(y, t) ** 2 + diff(z, t) ** 2)
    + moi_Q * diff(theta, t) ** 2
    + mass_P * (vyP**2 + vzP**2)
)

# Potential energy
V = g * mass_Q * z + g * mass_P * (z - l * cos(phi))

# Input forces
f = Matrix([-sin(theta), cos(theta)])

r_1 = Matrix([y - a * cos(theta), z - a * sin(theta)]) 
r_2 = Matrix([y + a * cos(theta), z + a * sin(theta)]) 

W_1 = wrenchMatrix(r_1, q)
W_2 = wrenchMatrix(r_2, q)

B = (W_1 * f).row_join(W_2 * f)

# Quantities
M = massMatrix(T, q, t)
c = velocityTerms(T, q, t)
tau_p = potentialTerms(V, q)

# Printout
print("T &=", latex(simplify(T)), "\\\\")
print("V &=", latex(simplify(V)))
print("M &=", latex(simplify(M)), "\\\\")
print("c &=", latex(simplify(c)), "\\\\")
print("\\tau_p &=", latex(simplify(tau_p)), "\\\\")
print("B &=", latex(simplify(B)))
