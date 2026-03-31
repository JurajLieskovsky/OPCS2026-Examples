from sympy import symbols, Function, Matrix, diff, cos, sin, latex, simplify
from lagrangian2eom import massMatrix, velocityTerms, potentialTerms, wrenchMatrix

t = symbols("t")

y = Function("y")(t)
z = Function("z")(t)
theta = Function("theta")(t)
phi = Function("phi")(t)
l = Function("l")(t)

g, mass_Q, moi_Q, mass_P, a = symbols("g m_Q I_Q m_P a")

# Generalized coordinates
q = Matrix([y, z, theta, phi])

# Kinetic energy
vyP = diff(y, t) + l * diff(phi, t) * cos(phi) + diff(l,t) * sin(phi)
vzP = diff(z, t) + l * diff(phi, t) * sin(phi) - diff(l,t) * cos(phi)

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

# Input forces
k1, k2 = symbols("kappa_1 kappa_2")

d = Matrix([-1, 0])

r_3 = Matrix([y, z]) 
r_4 = Matrix([y + l * sin(theta), z - l * cos(theta)]) 

W_3 = wrenchMatrix(r_3, q)
W_4 = wrenchMatrix(r_4, q)

E = k1 * W_3 * d + k2 * W_4 * d

# Quantities
M = massMatrix(T, q, t)
c = velocityTerms(T, q, t) + diff(diff(T, diff(q, t).T), l).T * diff(l, t)
tau_p = potentialTerms(V, q)

# Printout
print("T &=", latex(simplify(T)), "\\\\")
print("V &=", latex(simplify(V)))
print("M &=", latex(simplify(M)), "\\\\")
print("c &=", latex(simplify(c)), "\\\\")
print("\\tau_p &=", latex(simplify(tau_p)), "\\\\")
print("B &=", latex(simplify(B)))
print("E &=", latex(simplify(E)))
