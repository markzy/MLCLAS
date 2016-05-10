from scipy.optimize import minimize

aa = [1, 2]


def opt(x, size):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def cons1(x, a, index):
    print(a, index)
    return x[0] - 2 * x[1] + 2


def cons2(x):
    return -x[0] - 2 * x[1] + 6


def cons3(x):
    return -x[0] + 2 * x[1] + 2


cons = [{'type': 'ineq', 'fun': cons1, 'args': (aa, 1)},
        {'type': 'ineq', 'fun': cons2},
        {'type': 'ineq', 'fun': cons3}]
# bnds = ((0, None), (0, None))
res = minimize(opt, [0, 0], args=((1, 2),), method='SLSQP', constraints=cons)
print(res)
