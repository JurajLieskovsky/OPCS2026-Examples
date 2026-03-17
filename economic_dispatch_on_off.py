import cvxpy as cp

a = [20, 25, 30]
b = [5, 4, 3]
c = [0.02, 0.15, 0.01]

l = [300, 200, 100]
u = [1000, 1500, 800]

x = cp.Variable(3)
z = cp.Variable(3, boolean=True)

objective = cp.Minimize(sum(a) + b @ x + c @ cp.square(x))
constraints = [
    cp.multiply(z, l) <= x,
    x <= cp.multiply(z, u),
    sum(x) == 1500
]

problem = cp.Problem(objective, constraints)
problem.solve(cp.SCIP)

print(objective.value)
print(x.value)
