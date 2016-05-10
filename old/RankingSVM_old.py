import numpy as np
import scipy.sparse
from scipy.optimize import minimize


def optifun(w, size):
    w_r = np.reshape(w, size)
    return np.sum(np.square(w_r[:, :(size[1] - 1)]))


def cons(w, size, x, indices):
    w_r = np.reshape(w, size)
    factors = w_r[indices]
    diff = np.array([factors[0] - factors[1]])
    return np.dot(diff,x.T)[0,0] - 1


def RSVMfit(X, y):
    # prepare the data
    if isinstance(X, scipy.sparse.spmatrix):
        X_array = X.toarray()
    else:
        X_array = np.array(X)
    y = np.array(y)
    samples, features = X.shape
    classes = y.shape[1]

    islabels = np.array([[j for j in range(classes) if y[i, j] == 1] for i in range(samples)])
    notlabels = np.array([[j for j in range(classes) if y[i, j] == 0] for i in range(samples)])

    append_feature = np.array([np.ones(samples)]).T
    X_array = np.concatenate((X_array, append_feature), axis=1)

    # convert the whole matrix into the vector
    initial_guess = np.zeros(classes * (features + 1))

    _constraints = []
    for index in range(samples):
        for label in islabels[index]:
            for nonlabel in notlabels[index]:
                _constraints.append({'type': 'ineq', 'fun': cons, 'args': ((classes, features + 1), np.array([X_array[index]]), np.array([label, nonlabel]))})

    res = minimize(optifun, initial_guess, args=((classes, features + 1),), method='SLSQP', constraints=_constraints)
    print(res)
