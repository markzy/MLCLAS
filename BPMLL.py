import math
import random
import numpy as np
import scipy
from scipy.sparse import vstack

from Models import BPMLL_models


# Backpropagation for Multi-Label Learning
class BPMLL:
    def __init__(self, neural=0.2, epoch=20, weight_decay=0, regulization=0, normalize=False, print_procedure=False):
        self.features = 0
        self.classes = 0
        self.samples = 0
        self.neural_num = 0
        self.normalize = normalize
        self.learn_rate = 0.05

        # these attributes affects the output
        self.neural_percent = neural
        self.epoch = epoch
        self.weightsDecayCost = weight_decay
        self.regulization = regulization

        self.error_small_change = 0.00001
        self.final_error = 0
        self.dataset = []
        self.threshold = None
        self.wsj_matrix = []
        self.vhs_matrix = []
        self.bias_b = []
        self.bias_a = []
        self.print_procedure = print_procedure

    def init(self, X, y):

        if isinstance(X, scipy.sparse.spmatrix):
            X_array = X.toarray()
        else:
            X_array = np.array(X)

        y = np.array(y)
        self.samples, self.features = X_array.shape
        self.classes = y.shape[1]

        self.dataset = self.prepare_data(X_array, y)
        self.samples = len(self.dataset)

        if self.samples < self.features:
            raise Exception("Your must have more instances than features")

        self.neural_num = int(self.features * self.neural_percent)

        self.wsj_matrix = np.array([[(random.random() - 0.5) for j in range(self.classes)] for s in range(self.neural_num)])
        self.vhs_matrix = np.array([[(random.random() - 0.5) for s in range(self.neural_num)] for h in range(self.features)])

        self.bias_b = np.ones((1, self.classes))
        self.bias_a = np.ones((1, self.neural_num))

        # return self

    def fit(self, X, y):
        self.init(X, y)
        self.iterate_training()
        return self

    def prepare_data(self, X_array, y):
        dataset = []

        if self.normalize is True:
            X_array = BPMLL_models.Nomalizer(X_array, -1, 1).normalize()

        for i in range(self.samples):
            # skip samples whose Yi or n-Yi is an empty set
            if np.sum(y) != 0 and np.sum(y) != self.classes:
                dataset.append(BPMLL_models.TrainPair(X_array[i], y[i]))

        return dataset

    def iterate_training(self):
        prev_error = self.global_error()
        for ep in range(self.epoch):
            if self.print_procedure:
                print("entering epoch " + str(ep))
            random.shuffle(self.dataset)

            for i in range(self.samples):
                self.fit_once(i)

            error = self.global_error()
            diff = prev_error - error
            if diff <= self.error_small_change * prev_error:
                self.build_threshhold()
                self.final_error = error
                return
            prev_error = error

        self.build_threshhold()
        self.final_error = prev_error
        return

    def fit_once(self, index):
        x = self.dataset[index].attributes
        x_vec = np.array([x]).T
        y = self.dataset[index].labels

        isLabel = self.dataset[index].isLabel
        notLabel = self.dataset[index].notLabel
        isLabel_length = len(isLabel)
        notLabel_length = len(notLabel)

        b, c = self.forward_propagation(x)

        exp_func = math.exp
        dj_sigma = np.zeros((1, self.classes))
        for j in range(self.classes):
            tmp = 0
            if y[j] == 1:
                for l in notLabel:
                    tmp += exp_func(-(c[0, j] - c[0, l]))
            else:
                for k in isLabel:
                    tmp -= exp_func(-(c[0, k] - c[0, j]))
            dj_sigma[0, j] = tmp

        d = (1 / (isLabel_length * notLabel_length)) * dj_sigma * (1 - np.square(c))

        # compute e matrix
        b_vec = b.T
        d_vec = d.T
        es_sigma_vec = np.dot(self.wsj_matrix, d_vec)
        e_vec = es_sigma_vec * (1 - np.square(b_vec))

        self.wsj_matrix = (1 - self.weightsDecayCost) * self.wsj_matrix + self.learn_rate * np.dot(b_vec, d)

        e = e_vec.T
        self.vhs_matrix = (1 - self.weightsDecayCost) * self.vhs_matrix + self.learn_rate * np.dot(x_vec, e)

        self.bias_b = (1 - self.weightsDecayCost) * self.bias_b + self.learn_rate * d
        self.bias_a = (1 - self.weightsDecayCost) * self.bias_a + self.learn_rate * e

        return

    def forward_propagation(self, x):
        x = np.array([x])

        ac_func = BPMLL_models.ActivationFunction().activate
        netb = np.dot(x, self.vhs_matrix) + self.bias_a
        b = ac_func(netb)

        netc = np.dot(b, self.wsj_matrix) + self.bias_b
        c = ac_func(netc)

        return b, c

    def global_error(self):
        global_error = 0

        weights_square_sum = np.sum(np.square(self.wsj_matrix)) + np.sum(
                np.square(self.vhs_matrix)) + np.sum(np.square(self.bias_b)) + np.sum(np.square(self.bias_a))

        exp_func = math.exp

        for i in range(self.samples):
            c = self.forward_propagation(self.dataset[i].attributes)[1]

            yi = self.dataset[i].isLabel
            nyi = self.dataset[i].notLabel
            yi_length = len(yi)
            nyi_length = len(nyi)

            A = np.array([[c[0, l] - c[0, k] for k in yi] for l in nyi])
            global_error += 1 / (yi_length * nyi_length) * np.sum(np.exp(A))

        global_error += self.regulization * weights_square_sum
        return global_error

    def build_threshhold(self):
        modelOutputs = []
        idealLabels = []
        for i in range(self.samples):
            c = self.forward_propagation(self.dataset[i].attributes)[1][0]
            modelOutputs.append(c)
            idealLabels.append(self.dataset[i].labels)

        self.threshold = BPMLL_models.ThresholdFunction(modelOutputs, idealLabels)

    def predict(self, X):
        samples, features = X.shape
        if features != self.features:
            raise Exception("inconsistent feature dimension")

        if isinstance(X, scipy.sparse.spmatrix):
            X_array = X.toarray()
        else:
            X_array = np.array(X)

        if self.normalize is True:
            X_array = BPMLL_models.Nomalizer(X_array, -1, 1).normalize()

        result = BPMLL_models.BPMLLResults(self.final_error)
        for i in range(samples):
            sample_result = []
            topLabel = None
            c = self.forward_propagation(X_array[i])[1][0]
            max_value = 0
            threshold = self.threshold.computeThreshold(c)
            for j in range(self.classes):
                if c[j] >= threshold:
                    sample_result.append(j)
                if c[j] > max_value:
                    topLabel = j
                    max_value = c[j]
            result.add(sample_result, topLabel, c)

        return result
