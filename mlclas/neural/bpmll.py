import math
import numpy as np
import random
from mlclas.neural import bpmll_models
from mlclas.utils import check_feature_input, check_target_input
from mlclas.stats import Normalizer, RankResults


class BPMLL:
    """
    Backpropagation for Multi-Label Learning based on:
    >   Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization."
        Knowledge and Data Engineering, IEEE Transactions on 18.10 (2006): 1338-1351.

    Init Parameters
    ----------
    neural : float, (default=0.2)
        decide the number of neurals in the hidden layer of the network, which equals feature_number * neural

    epoch : int, (default=20)
        decide the maximum number of training epochs, the training process will terminate when it is reached

    weight_decay : float, (deafault=0.00001)
        the parameter used to decay the value of parameters during each training, will may prevent over-fitting

    regularization : float, (default=0.1)
        the parameter used to calculate the global error term, which may prevent over-fitting

    normalize: bool, (default=False)
        decide whether and how to normalize the input array

    print_procedure:
        decide whether print the middle status of the training process to the std output

    """

    def __init__(self, neural=0.2, epoch=20, weight_decay=0.00001, regularization=0.1, normalize=False, axis=0, print_procedure=False):
        self.features = 0
        self.classes = 0
        self.samples = 0
        self.neural_num = 0

        self.normalize = normalize
        self.axis = axis
        self.learn_rate = 0.05

        # these attributes affects the output
        self.neural_percent = neural
        self.epoch = epoch
        self.weightsDecayCost = weight_decay
        self.regularization = regularization

        # attributes used by methods
        self.error_small_change = 0.00001
        self.final_error = 0
        self.dataset = []
        self.threshold = None
        self.wsj_matrix = []
        self.vhs_matrix = []
        self.bias_b = []
        self.bias_a = []
        self.print_procedure = print_procedure

        self.trained = False

    def fit(self, x, y):
        """Fit underlying estimators.

        Parameters
        ----------
        x : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-Label targets.

        Returns
        -------
        self
        """

        # if self.trained is True:
        #     raise Exception('this classifier has already been trained, please create a new classifier')

        x = check_feature_input(x)
        y = check_target_input(y)

        self.features = x.shape[1]
        self.classes = y.shape[1]

        self.dataset = self.prepare_data(x, y)

        self.samples = len(self.dataset)

        if self.samples < self.features:
            raise Exception("Your must have more instances than features")

        self.neural_num = int(self.features * self.neural_percent)

        # weights between the hidden layer and the output layer
        self.wsj_matrix = np.random.random_sample((self.neural_num, self.classes)) - 0.5
        # weights between the input layer and the hidden layer
        self.vhs_matrix = np.random.random_sample((self.features, self.neural_num)) - 0.5

        # bias between the hidden layer and the output layer
        self.bias_b = np.ones((1, self.classes))
        # bias between the input layer and the hidden layer
        self.bias_a = np.ones((1, self.neural_num))

        self.iterate_training()

        self.trained = True
        return self

    def prepare_data(self, x, y):
        dataset = []

        x = Normalizer.normalize(x, norm=self.normalize, axis=self.axis)

        for i in range(x.shape[0]):
            # skip samples whose Yi or n-Yi is an empty set
            if np.sum(y) != 0 and np.sum(y) != self.classes:
                dataset.append(bpmll_models.TrainPair(x[i], y[i]))

        return dataset

    def iterate_training(self):
        """ iterate the training process until converge """
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

        if self.print_procedure:
            print('exceed the maximum epochs, training process terminated')

        self.build_threshhold()
        self.final_error = prev_error
        return

    def fit_once(self, index):
        x = self.dataset[index].attributes
        y = self.dataset[index].labels
        x_vec = np.array([x]).T

        is_label = self.dataset[index].isLabel
        not_label = self.dataset[index].notLabel
        is_label_length = len(is_label)
        not_label_length = len(not_label)

        b, c = self.forward_propagation(x)

        exp_func = math.exp
        dj_sigma = np.zeros((1, self.classes))
        for j in range(self.classes):
            tmp = 0
            if y[j] == 1:
                for l in not_label:
                    tmp += exp_func(-(c[0, j] - c[0, l]))
            else:
                for k in is_label:
                    tmp -= exp_func(-(c[0, k] - c[0, j]))
            dj_sigma[0, j] = tmp

        # general error of the output layer
        d = (1 / (is_label_length * not_label_length)) * dj_sigma * (1 - np.square(c))

        # compute general error of the hidden layer e
        b_vec = b.T
        d_vec = d.T

        es_sigma_vec = np.dot(self.wsj_matrix, d_vec)
        e_vec = es_sigma_vec * (1 - np.square(b_vec))
        e = e_vec.T

        # update weights and biases
        self.wsj_matrix = (1 - self.weightsDecayCost) * self.wsj_matrix + self.learn_rate * np.dot(b_vec, d)
        self.vhs_matrix = (1 - self.weightsDecayCost) * self.vhs_matrix + self.learn_rate * np.dot(x_vec, e)

        self.bias_b = (1 - self.weightsDecayCost) * self.bias_b + self.learn_rate * d
        self.bias_a = (1 - self.weightsDecayCost) * self.bias_a + self.learn_rate * e

        return

    def forward_propagation(self, x):
        x = np.array([x])

        netb = np.dot(x, self.vhs_matrix) + self.bias_a
        b = bpmll_models.ActivationFunction.activate(netb)

        netc = np.dot(b, self.wsj_matrix) + self.bias_b
        c = bpmll_models.ActivationFunction.activate(netc)

        return b, c

    def global_error(self):
        global_error = 0

        weights_square_sum = np.sum(np.square(self.wsj_matrix)) + np.sum(
                np.square(self.vhs_matrix)) + np.sum(np.square(self.bias_b)) + np.sum(np.square(self.bias_a))

        for i in range(self.samples):
            c = self.forward_propagation(self.dataset[i].attributes)[1]

            yi = self.dataset[i].isLabel
            nyi = self.dataset[i].notLabel
            yi_length = len(yi)
            nyi_length = len(nyi)

            A = np.array([[c[0, l] - c[0, k] for k in yi] for l in nyi])
            global_error += 1 / (yi_length * nyi_length) * np.sum(np.exp(A))

        global_error += self.regularization * weights_square_sum
        return global_error

    def build_threshhold(self):
        model_outputs = []
        ideal_labels = []
        for i in range(self.samples):
            c = self.forward_propagation(self.dataset[i].attributes)[1][0]
            model_outputs.append(c)
            ideal_labels.append(self.dataset[i].labels)

        self.threshold = bpmll_models.ThresholdFunction(model_outputs, ideal_labels)

    def predict(self, x, rank_results=False):
        """
        predict process
        :param x: feature matrix
        :param rank_results: decide whether return RankResults object or the actual output
        :return: result: RankResults
        """
        if self.trained is False:
            raise Exception('this classifier has not been trained')

        x = check_feature_input(x)
        samples, features = x.shape

        if features != self.features:
            raise Exception("inconsistent feature dimension")

        x = Normalizer.normalize(x, norm=self.normalize, axis=self.axis)

        result = RankResults()

        for sample_index in range(samples):
            sample_result = []

            c = self.forward_propagation(x[sample_index])[1][0]
            threshold = self.threshold.compute_threshold(c)

            top_label = None
            max_value = 0
            count = 0
            for j in range(self.classes):
                if c[j] >= threshold:
                    count += 1
                    sample_result.append(j)
                if c[j] > max_value:
                    top_label = j
                    max_value = c[j]

            # append the top label if no label satisfies the threshold value
            if count == 0:
                sample_result.append(top_label)

            result.add(sample_result, top_label, c)

        if rank_results is False:
            result = result.predictedLabels

        return result
