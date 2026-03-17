import cvxpy as cp

x = cp.Variable(2, integer=True)
objective = cp.Maximize(0.25 * x[0] + 0.0075 * x[1])
constraints = [
    x >= 0,
    cp.sum(x) <= 60,
    2.5 * x[0] + 0.5 * x[1] <= 42,
]
problem = cp.Problem(objective, constraints)

result = problem.solve()

print(objective.value)
print(x.value)
