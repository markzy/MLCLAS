import numpy as np
import cvxpy as cvx
import operator
from mlclas.svm.rankingsvm_models import *
from mlclas.neural.bpmll_models import ThresholdFunction
from mlclas.utils import check_feature_input, check_target_input
from mlclas.stats import Normalizer, RankResults


class RankingSVM:
    """
    RankingSVM algorithm based on:
    >   Elisseeff, André, and Jason Weston.
        "Kernel methods for Multi-labelled classification and Categorical regression problems"
        Advances in neural information processing systems. 2001.

    Init Parameters
    ----------
    print_procedure:
    decide whether print the middle status of the training process to the std output
    """

    def __init__(self, normalize=False, axis=0, print_procedure=False):
        self.w = None
        self.threshold = None
        self.normalize = normalize
        self.axis = axis
        self.print_procedure = print_procedure
        self.trained = False

    def fit(self, x, y, c_factor):
        x = check_feature_input(x)
        y = check_target_input(y)

        self.features = x.shape[1]

        x = Normalizer.normalize(x, self.normalize, self.axis)

        sample_num, feature_num = x.shape
        class_num = y.shape[1]
        class_info = AllLabelInfo()

        """ Franke and Wolfe Method applied on the optimization problem """

        # organize labels for preparation
        for sample_index in range(sample_num):
            sample_y = y[sample_index]
            labels = []
            not_labels = []
            for label_index in range(class_num):
                if sample_y[label_index] == 1:
                    labels.append(label_index)
                else:
                    not_labels.append(label_index)
            class_info.append(labels, not_labels)

        """
        Alpha ought to have 3-dimensions. For convenience, it is flattened into a list.
        It's length is sum(yi*nyi) for i in range(samle_num)
        """
        alpha = np.zeros(class_info.totalProduct)
        alpha_var = cvx.Variable(class_info.totalProduct)
        """
        C has 4 dimensions, which is hard to present.
        In this program it has 2 dimensions, which is i(sample index) and k(class index).
        Each c[i][k] contains a yi*nyi array, which is initialized according to the original paper.
        """
        c = [[0 for k in range(class_num)] for i in range(sample_num)]
        for i in range(sample_num):
            sample_shape, labels, not_labels = class_info.get_shape(i, True)
            for k in range(class_num):
                matrix = np.zeros(sample_shape)
                if k in labels:
                    index = labels.index(k)
                    matrix[index, :] = 1
                else:
                    index = not_labels.index(k)
                    matrix[:, index] = -1
                c[i][k] = matrix.flatten()
        c = np.array(c)

        beta = np.zeros((class_num, sample_num))
        beta_new = np.zeros((class_num, sample_num))
        wx_inner = np.zeros((class_num, sample_num))

        # TODO: this can cut half of the running time
        x_inner = np.array([[np.inner(x[i], x[j]) for j in range(sample_num)] for i in range(sample_num)])
        g_ikl = np.zeros(class_info.totalProduct)

        """ prepare for the first linear programming """
        c_i = class_info.eachProduct
        bnds = []
        for i in range(sample_num):
            bnds += [c_factor / c_i[i] for j in range(c_i[i])]
        bnds = np.array(bnds)

        zeros = np.zeros(class_info.totalProduct)
        zeros.fill(1e-5)
        A_lp = []
        for k in range(1, class_num):
            A_lp.append(np.concatenate(c[:, k]).tolist())
        A_lp = np.array(A_lp)
        b_lp = np.zeros(class_num - 1)

        cons = [zeros <= alpha_var, alpha_var <= bnds, A_lp * alpha_var == b_lp]

        converge = False
        iteration_count = 0
        # iterate training until converge
        while not converge:
            iteration_count += 1
            # compute beta
            for i in range(sample_num):
                alpha_range = class_info.get_range(i)
                alpha_piece = alpha[alpha_range[0]:alpha_range[1]]
                c_list = c[i]
                for k in range(class_num):
                    beta[k][i] = np.inner(c_list[k], alpha_piece)

            # compute <w_k, x_j>
            for k in range(class_num):
                beta_list = beta[k]
                for j in range(sample_num):
                    x_innerlist = x_inner[:, j]
                    wx_inner[k][j] = np.inner(beta_list, x_innerlist)

            # compute g_ikl
            for i in range(sample_num):
                g_range = class_info.get_range(i)
                shape, labels, not_labels = class_info.get_shape(i, True)
                wx_list = wx_inner[:, i]
                g_ikl[g_range[0]:g_range[1]] = np.repeat(wx_list[labels], shape[1]) - np.tile(wx_list[not_labels], shape[0]) - 1

            """
            optimization problem 1:
            solve min<g, alpha_new> with corresponding constraints
            """

            if self.print_procedure:
                print('iteration %d...' % iteration_count)

            """
            cvxopt.solvers.lp Solves a pair of primal and dual LPs

            minimize    c'*x
            subject to  G*x + s = h
                        A*x = b
                        s >= 0

            maximize    -h'*z - b'*y
            subject to  G'*z + A'*y + c = 0
                        z >= 0.

            cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])
            """

            obj = cvx.Minimize(cvx.sum_entries(g_ikl * alpha_var))
            prob = cvx.Problem(obj, cons)
            prob.solve()

            alpha_new = np.array(alpha_var.value).T[0]

            """ now the problem collapse into a really simple qp problem """
            # compute beta_new
            for i in range(sample_num):
                alpha_range = class_info.get_range(i)
                alpha_piece = alpha_new[alpha_range[0]:alpha_range[1]]
                c_list = c[i]
                for k in range(class_num):
                    beta_new[k][i] = np.inner(c_list[k], alpha_piece)

            """
            We need to find lambda which will make W(alpha + lambda*alpha_new) be maximum
            and alpha + lambda*alpha_new satisfies the previous constraints.

            After calculating the formula, it is now:
            new_W = old_W + [sum formula of beta_new and beta] + [sum formula of beta and beta_new]
            + [sum formula of beta_new and beta_new] + [sum of the new alpha]

            old_W is fixed and has no effect on the choice of the lambda during the maximum process.
            Then we can calculate the coeffient of lambda. The final problem will look like:

            maximize    a*lambda_square + b*lambda
                        c <= lambda <= d

            It is apparently easy to solve.
            """
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
            for alpha_index in range(class_info.totalProduct):
                if not alpha_new[alpha_index] == 0:
                    left = max(left_vec[alpha_index] / alpha_new[alpha_index], left)
                    right = min(right_vec[alpha_index] / alpha_new[alpha_index], right)

            optifunc = lambda z: lambda_2 * z * z + lambda_1 * z

            # decide lambda's value
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

            if self.print_procedure:
                print("final lambda: " + str(final_lambda))
                print("optifunc: " + str(optifunc(final_lambda)))

            # converge condition
            if optifunc(final_lambda) <= 1 or final_lambda <= 1e-3:
                converge = True
            else:
                alpha += final_lambda * alpha_new

        """ compute w and b via KKT conditions """
        w = [0 for i in range(class_num)]
        for k in range(class_num):
            beta_vec = np.asarray([beta[k]])
            w[k] = beta_vec.dot(x)[0]

        w = np.array(w)
        b = np.zeros(class_num)

        # use x[0] to compute differences of b
        x_list = x[0]
        shape, labels, not_labels = class_info.get_shape(0, True)

        """
        We know that once the differences between each element and the first element is known
        we can get all the values of the list.
        As the classification system is based on ranking, we can make any element 0 as a start of calculation,
        which will not affect the final ranking.
        """
        # make the first label's b=0, it won't affect the fianl ranking
        for l in not_labels:
            b[l] = np.inner(w[labels[0]] - w[l], x_list) - 1

        # then use b[falselabels[0]] to compute b[actuallabels[1:]]
        falselabelb = b[not_labels[0]]
        falselabel_index = not_labels[0]
        for labelIndex in range(1, len(labels)):
            b[labels[labelIndex]] = 1 + falselabelb - np.inner(w[labels[labelIndex]] - w[falselabel_index], x_list)

        # build threshold for labeling
        x_extend = np.concatenate((x, np.array([np.ones(sample_num)]).T), axis=1)
        w_extend = np.concatenate((w, np.array([b]).T), axis=1)
        model_outputs = np.dot(x_extend, w_extend.T)
        self.threshold = ThresholdFunction(model_outputs, y)
        self.w = w_extend

        self.trained = True
        return self

    def predict(self, x, rank_results=False):
        if self.trained is False:
            raise Exception('this classifier has not been trained')

        x = check_feature_input(x)
        x = Normalizer.normalize(x, self.normalize, self.axis)
        sample_num, feature_num = x.shape
        class_num = self.w.shape[0]

        if feature_num != self.w.shape[1] - 1:
            raise Exception('testing samples have inconsistent shape of training samples!')

        x_extend = np.concatenate((x, np.array([np.ones(sample_num)]).T), axis=1)
        threshold = self.threshold
        outputs = np.dot(x_extend, self.w.T)

        result = RankResults()

        for index in range(sample_num):
            sample_result = []
            op = outputs[index]
            th = threshold.compute_threshold(op)

            top_label = None
            max_value = float('-inf')
            count = 0
            for j in range(class_num):
                if op[j] >= th:
                    count += 1
                    sample_result.append(j)
                if op[j] > max_value:
                    top_label = j
                    max_value = op[j]

            if count == 0:
                sample_result.append(top_label)

            result.add(sample_result, top_label, op)

        if rank_results is False:
            result = result.predictedLabels

        return result
