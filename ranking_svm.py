import numpy as np
import scipy.sparse
import cvxopt as ct
import pickle
from Models.RankingSVM_models import *
from Models.BPMLL_models import ThresholdFunction
import operator


class RankingSVM:
    def __init__(self, w, threshold):
        self.w = w
        self.threshold = threshold


def predict(X, w, threshold):
    if isinstance(X, scipy.sparse.spmatrix):
        X_array = X.toarray()
    else:
        X_array = np.array(X)
    sample_num, feature_num = X.shape
    class_num = w.shape[0]

    if feature_num != w.shape[1] - 1:
        raise Exception('inconsistent shape of training samples!')

    X_extend = np.concatenate((X_array, np.array([np.ones(sample_num)]).T), axis=1)

    outputs = np.dot(X_extend, w.T)
    result = []
    for index in range(sample_num):
        sample_result = []
        op = outputs[index]
        th = threshold.computeThreshold(op)
        count = 0
        for j in range(class_num):
            if op[j] >= th:
                count += 1
                sample_result.append(j)
        if count == 0:
            op_index, op_value = max(enumerate(op), key=operator.itemgetter(1))
            for j in range(class_num):
                if op[j] == op_value:
                    sample_result.append(j)
        result.append(sample_result)
    return result


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

    # iterate training until converge
    beta = np.zeros((class_num, sample_num))
    beta_new = np.zeros((class_num, sample_num))
    wx_inner = np.zeros((class_num, sample_num))

    # !! this can cut half of the running time
    x_inner = np.array([[np.inner(X_array[i], X_array[j]) for j in range(sample_num)] for i in range(sample_num)])
    g_ikl = np.zeros(classInfo.totalProduct)

    """ prepare for the first linear programming """
    c_i = classInfo.eachProduct
    bnds = []
    for i in range(sample_num):
        bnds += [C_factor / c_i[i] for j in range(c_i[i])]

    G_lp = ct.matrix(np.concatenate([-np.eye(classInfo.totalProduct), np.eye(classInfo.totalProduct)]))
    h_lp = ct.matrix(np.concatenate([np.zeros(classInfo.totalProduct), np.array(bnds)]))
    A_lp = []
    for k in range(1, class_num):
        A_lp.append(np.concatenate(c[:, k]).tolist())
    A_lp = ct.matrix(np.array(A_lp))
    b_lp = ct.matrix(np.zeros(class_num - 1))

    converge = False
    iteration_count = 0

    while not converge:
        iteration_count += 1
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

        # finally trying to ues cvxopt
        c_lp = ct.matrix(g_ikl)
        print('iteration %d...' % iteration_count)
        sol = ct.solvers.lp(c_lp, G_lp, h_lp, A_lp, b_lp)
        sol_matrix = sol['x']

        # some code used for persisting middle results, for test usages
        # file_name = '/Users/Mark/PycharmProjects/multi_label_classification/results/SVM/RSVM.pkl'
        # with open(file_name, 'wb') as op:
        #     pickle.dump(sol['x'], op, pickle.HIGHEST_PROTOCOL)
        # exit()
        # with open(file_name, 'rb') as ip:
        #     sol_matrix = pickle.load(ip)

        alpha_new = np.array(sol_matrix).T[0]

        # now the problem collapse into a simple lp problem
        # compute beta_new
        for i in range(sample_num):
            alpha_range = classInfo.getRangeFromIndex(i)
            alpha_piece = alpha_new[alpha_range[0]:alpha_range[1]]
            c_list = c[i]
            for k in range(class_num):
                beta_new[k][i] = np.inner(c_list[k], alpha_piece)

        # init coeffient of lambda
        lambda_11 = np.sum(beta_new.T.dot(beta) * x_inner)
        lambda_12 = np.sum(beta.T.dot(beta_new) * x_inner)
        lambda_13 = np.sum(alpha_new)
        # coefficient of lambda
        lambda_1 = lambda_13 - lambda_11 / 2 - lambda_12 / 2
        # coefficient of lambda square
        lambda_2 = np.sum(beta_new.T.dot(beta_new) * x_inner) / (-2)

        # prepare constraints
        left_vec = - alpha
        right_vec = bnds - alpha
        left = float('-inf')
        right = float('inf')
        for alpha_index in range(classInfo.totalProduct):
            if not alpha_new[alpha_index] == 0:
                left = max(left_vec[alpha_index] / alpha_new[alpha_index], left)
                right = min(right_vec[alpha_index] / alpha_new[alpha_index], right)

        optifunc = lambda x: lambda_2 * x * x + lambda_1 * x

        # decide lambda's value
        final_lambda = 0
        if lambda_2 < 0:
            opti_lambda = -lambda_1 / (lambda_2 * 2)
            if opti_lambda < left:
                final_lambda = left
            elif opti_lambda > right:
                final_lambda = right
            else:
                final_lambda = opti_lambda
        elif lambda_2 == 0:
            if lambda_1 >= 0:
                final_lambda = right
            else:
                final_lambda = left
        else:
            worst_lambda = -lambda_1 / (lambda_2 * 2)
            if worst_lambda < left:
                final_lambda = right
            elif worst_lambda > right:
                final_lambda = left
            else:
                final_lambda = left if optifunc(left) >= optifunc(right) else right

        print("final lambda: " + str(final_lambda))
        print("optifunc: " + str(optifunc(final_lambda)))

        # converge condition
        if optifunc(final_lambda) <= 1 or final_lambda <= 1e-3:
            converge = True
        else:
            alpha += final_lambda * alpha_new

    with open('results/SVM/alpha.pkl', 'wb') as _input:
        pickle.dump(alpha, _input, pickle.HIGHEST_PROTOCOL)
        print('successfully preserved final alpha')

    # compute w&b via KKT conditions
    w = [0 for i in range(class_num)]
    for k in range(class_num):
        beta_vec = np.asarray([beta[k]])
        w[k] = beta_vec.dot(X_array)[0]

    w = np.array(w)
    b = np.zeros(class_num)

    # use x[0] to compute differences of b
    x_list = X_array[0]
    shape, labels, notLabels = classInfo.getShape(0, True)

    # make the first label's b=0, it won't affect the fianl ranking
    for l in notLabels:
        b[l] = np.inner(w[labels[0]] - w[l], x_list) - 1

    # then use b[falselabels[0]] to compute b[actuallabels[1:]]
    falselabelb = b[notLabels[0]]
    falselabelIndex = notLabels[0]
    for labelIndex in range(1, len(labels)):
        b[labels[labelIndex]] = 1 + falselabelb - np.inner(w[labels[labelIndex]] - w[falselabelIndex], x_list)

    # build threshold for labeling
    X_extend = np.concatenate((X_array, np.array([np.ones(sample_num)]).T), axis=1)
    w_extend = np.concatenate((w, np.array([b]).T), axis=1)
    model_outputs = np.dot(X_extend, w_extend.T)
    threshold_function = ThresholdFunction(model_outputs, y)
    result = RankingSVM(w_extend, threshold_function)

    return result
