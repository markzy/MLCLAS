import numpy as np
import scipy.sparse
from Models.RankingSVM_models import *
from scipy.optimize import linprog
import cvxopt as ct


# def optifun(alpha_new, g):
#     return np.inner(alpha_new, g)
#
#
# def cons(alpha_new, c_k):
#     return np.inner(alpha_new, c_k)
#


def fitRSVM(X, y, C_factor):
    if isinstance(X, scipy.sparse.spmatrix):
        X_array = X.toarray()
    else:
        X_array = np.array(X)
    y = np.array(y)

    sample_num, feature_num = X.shape
    class_num = y.shape[1]
    classInfo = AllLabelInfo()

    """ Franke and Wolfe Method applied on the optimization problem """

    # organize labels for preparation
    for sample_index in range(sample_num):
        sample_y = y[sample_index]
        labels = []
        notLabels = []
        for label_index in range(class_num):
            if sample_y[label_index] == 1:
                labels.append(label_index)
            else:
                notLabels.append(label_index)
        classInfo.append(labels, notLabels)

    # initialize alpha
    alpha = np.zeros(classInfo.totalProduct)

    # initialize c
    c = [[0 for k in range(class_num)] for i in range(sample_num)]
    for i in range(sample_num):
        sample_shape, labels, notLabels = classInfo.getShape(i, True)
        for k in range(class_num):
            matrix = np.zeros(sample_shape)
            if k in labels:
                index = labels.index(k)
                matrix[index, :] = 1
            else:
                index = notLabels.index(k)
                matrix[:, index] = -1
            c[i][k] = matrix.flatten()
    c = np.array(c)
    zzz = np.concatenate(c[:, 1])

    # iterate training until converge
    beta = np.array([[0 for i in range(sample_num)] for k in range(class_num)])
    wx_inner = np.array([[0 for i in range(sample_num)] for k in range(class_num)])

    # !! this can cut half of the running time
    x_inner = np.array([[np.inner(X_array[i], X_array[j]) for j in range(sample_num)] for i in range(sample_num)])

    g_ikl = np.zeros(classInfo.totalProduct)

    converge = False
    while not converge:
        # compute beta
        for i in range(sample_num):
            alpha_range = classInfo.getRangeFromIndex(i)
            alpha_piece = alpha[alpha_range[0]:alpha_range[1]]
            c_list = c[i]
            for k in range(class_num):
                beta[k][i] = np.inner(c_list[k], alpha_piece)

        # compute <w_k, x_j>
        for k in range(class_num):
            beta_list = beta[k]
            for j in range(sample_num):
                x_innerList = x_inner[:, j]
                wx_inner[k][j] = np.inner(beta_list, x_innerList)

        # compute g_ikl
        for i in range(sample_num):
            g_range = classInfo.getRangeFromIndex(i)
            shape, labels, notLabels = classInfo.getShape(i, True)
            wx_list = wx_inner[:, i]
            g_ikl[g_range[0]:g_range[1]] = np.repeat(wx_list[labels], shape[1]) - np.tile(wx_list[notLabels], shape[0]) - 1

        """
        optimization problem 1:
        solve min<g, alpha_new> with corresponding constraints
        """

        """ trying to use scipy minimize"""
        # alpha_new = np.zeros(classInfo.totalProduct)
        # bnds = [(0, None)] * classInfo.totalProduct
        # cts = [{'type': 'eq', 'fun': cons, 'args': (np.concatenate(c[:, k]),)} for k in range(class_num)]
        # res = minimize(optifun, alpha_new, args=(g_ikl,), method='SLSQP', bounds=bnds, constraints=cts)

        """ trying to ues cvxopt """
        c_i = classInfo.eachProduct
        bnds = []
        for i in range(sample_num):
            bnds += [C_factor / c_i[i] for j in range(c_i[i])]

        c_lp = ct.matrix(g_ikl)
        G_lp = ct.matrix(np.concatenate([-np.eye(classInfo.totalProduct), np.eye(classInfo.totalProduct)]))
        h_lp = ct.matrix(np.concatenate([np.zeros(classInfo.totalProduct), np.array(bnds)]))

        A_lp = []
        for k in range(1, class_num):
            A_lp.append(np.concatenate(c[:, k]).tolist())
        A_lp = ct.matrix(np.array(A_lp))
        b_lp = ct.matrix(np.zeros(class_num - 1))

        print('begin solving...')
        sol = ct.solvers.lp(c_lp, G_lp, h_lp, A_lp, b_lp)
        print(sol)
        exit()

        """ trying to use scipy linprog"""
        # A_lg = np.asarray([np.concatenate(c[:, k]).tolist() for k in range(1, class_num)])
        # b_lg = np.zeros(class_num - 1)
        #
        # c_i = classInfo.eachProduct
        # bnds = []
        # for i in range(sample_num):
        #     bnds += [(0, C_factor / c_i[i]) for j in range(c_i[i])]
        #
        # res = linprog(g_ikl,A_eq=A_lg,b_eq=b_lg,bounds=bnds)
