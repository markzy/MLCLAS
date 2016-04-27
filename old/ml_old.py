import math
import random

import numpy as np
import scipy
from scipy.sparse import vstack

from Models import BPMLL_models


# to-do: stop criterion can be refined
class MLDecisionTree:
    def __init__(self):
        self.features = 0
        self.classes = 0
        self.stop_criterion = 0
        self.root = None
        self.leaf_labels = []
        self.leaf_index = 0
        self.mode = None
        self.round = 0

    def fit(self, X, y, mode='gain', stop_criterion=10, round_val=1):

        # make sure the attributes are expected
        if not isinstance(X, scipy.sparse.spmatrix):
            raise Exception("Please make sure your instance space is a type of scipy sparse matrix, You can visit scipy.org for more information")

        if mode not in ['ratio', 'gain']:
            raise Exception("Unknown mode: " + mode)

        y = np.array(y)
        if len(y.shape) < 2:
            raise Exception("Please make sure your label space is in a binary form")

        # reinitialize the attributes, making the object refitable
        self.mode = mode
        self.features = X.shape[1]
        self.classes = y.shape[1]
        self.stop_criterion = stop_criterion
        self.leaf_labels = []
        self.leaf_index = 0
        self.round = round_val
        self.root = BPMLL_models.TreeNode()
        # fit tree recursively
        self.fit_tree(X, y, self.root)
        print(self.leaf_labels)
        return self

    def fit_tree(self, X, y, tree_node):

        mlent_all = self.get_mlent(y)

        sample_num = X.shape[0]
        # when stop criterion is met, store information in the leaf node
        if sample_num <= self.stop_criterion or mlent_all == 0:
            labels = []
            for i in range(self.classes):
                if np.sum(y[:, i]) / sample_num > 0.5:
                    labels.append(i)

            tree_node.data = self.leaf_index
            self.leaf_labels.append(labels)
            self.leaf_index += 1
            return

        max_feature_index, max_feature_dic = self.get_best_feature(X, y)
        split_value, left_indices, right_indices = self.get_best_split_value(max_feature_dic, y)

        # prepare the data and results for the next split
        tree_node.data = [max_feature_index, split_value]

        X_left = X.getrow(left_indices[0])
        y_left = np.array([y[left_indices[0]]])
        X_right = X.getrow(right_indices[0])
        y_right = np.array([y[right_indices[0]]])
        for index in range(1, len(left_indices)):
            X_left = vstack([X_left, X.getrow(left_indices[index])])
            y_left = np.concatenate((y_left, [y[left_indices[index]]]), axis=0)
        for index in range(1, len(right_indices)):
            X_right = vstack([X_right, X.getrow(right_indices[index])])
            y_right = np.concatenate((y_right, [y[right_indices[index]]]), axis=0)

        tree_node.left = BPMLL_models.TreeNode()
        self.fit_tree(X_left, y_left, tree_node.left)
        tree_node.right = BPMLL_models.TreeNode()
        self.fit_tree(X_right, y_right, tree_node.right)

    def get_best_feature(self, X, y):
        mlent_all = self.get_mlent(y)
        sample_num = X.shape[0]
        max_feature_ig_ratio = 0
        max_feature_index = None
        max_feature_dic = None
        for feature_index in range(self.features):
            feature_ig = mlent_all
            raw_feature_values = X.getcol(feature_index).toarray().T[0].tolist()
            feature_values = [round(num, self.round) for num in raw_feature_values]
            feature_dic = {}
            for index in range(len(feature_values)):
                if feature_values[index] in feature_dic:
                    feature_dic[feature_values[index]].append(index)
                else:
                    feature_dic[feature_values[index]] = [index]

            feature_split = 0
            sorted_keys = sorted(feature_dic.keys())
            for key in sorted_keys:
                split_sample_indices = feature_dic[key]
                split_count = len(split_sample_indices)
                if self.mode == 'ratio':
                    feature_split -= split_count / sample_num * math.log(split_count / sample_num, 2)
                else:
                    feature_split = 1
                split_y = np.array([y[split_sample_indices[0]]])
                for index in range(1, split_count):
                    split_y = np.concatenate((split_y, [y[split_sample_indices[index]]]), axis=0)
                feature_ig -= split_count / sample_num * self.get_mlent(split_y)
            feature_ig_ratio = feature_ig / feature_split

            if feature_ig_ratio > max_feature_ig_ratio:
                max_feature_ig_ratio = feature_ig_ratio
                max_feature_index = feature_index
                max_feature_dic = feature_dic
        return max_feature_index, max_feature_dic

    def get_best_split_value(self, max_feature_dic, y):
        mlent_all = self.get_mlent(y)
        sample_num = y.shape[0]
        sorted_feature_values = []
        corresponding_indices = []
        max_sorted_keys = sorted(max_feature_dic.keys())
        for key in max_sorted_keys:
            sample_indices = max_feature_dic[key]
            for index in sample_indices:
                sorted_feature_values.append(key)
                corresponding_indices.append(index)

        last_y = np.array([])
        max_ig_ratio = 0
        split_value = None
        left_indices = None
        right_indices = None
        for i in range(1, len(corresponding_indices) - 1):
            if not np.array_equal(y[corresponding_indices[i]], last_y):
                last_y = y[corresponding_indices[i]]
                y_left_indices = corresponding_indices[0:i]
                y_right_indices = corresponding_indices[i:]
                y_left = np.array([y[y_left_indices[0]]])
                y_right = np.array([y[y_right_indices[0]]])
                for index in range(1, len(y_left_indices)):
                    y_left = np.concatenate((y_left, [y[y_left_indices[index]]]), axis=0)
                for index in range(1, len(y_right_indices)):
                    y_right = np.concatenate((y_right, [y[y_right_indices[index]]]), axis=0)
                ig = mlent_all - (y_left.shape[0] / sample_num * self.get_mlent(y_left) + y_right.shape[0] / sample_num * self.get_mlent(y_right))
                if self.mode == 'ratio':
                    split_div = -len(y_left_indices) / sample_num * math.log(len(y_left_indices) / sample_num, 2) - len(
                            y_right_indices) / sample_num * math.log(len(y_right_indices) / sample_num, 2)
                else:
                    split_div = 1
                ig_ratio = ig / split_div
                if ig_ratio > max_ig_ratio:
                    max_ig_ratio = ig_ratio
                    split_value = (sorted_feature_values[i - 1] + sorted_feature_values[i]) / 2
                    left_indices = y_left_indices
                    right_indices = y_right_indices

        return split_value, left_indices, right_indices

    def get_mlent(self, y):
        samples, classes = y.shape
        if samples == 1:
            return 0
        pj = [np.sum(y[:, i]) / samples for i in range(classes)]
        mlent = 0
        for j in range(classes):
            pjs = pj[j]
            if pjs == 0 or pjs == 1:
                continue
            mlent += -pjs * math.log(pjs, 2) - (1 - pjs) * math.log((1 - pjs), 2)
        return mlent

    def plot_tree(self, treenode):
        print(treenode.data)
        if treenode.left is not None:
            self.plot_tree(treenode.left)
        if treenode.right is not None:
            self.plot_tree(treenode.right)

    def predict(self, X):
        predict_num, predict_features = X.shape
        if predict_features != self.features:
            exit("inconsistent feature number")

        results = []
        for i in range(predict_num):
            raw_feature_values = X.getrow(i).toarray()[0]
            feature_values = [round(num, self.round) for num in raw_feature_values]
            treenode = self.root
            while isinstance(treenode.data, list):
                feature_index = treenode.data[0]
                feature_div_value = treenode.data[1]
                if feature_values[feature_index] <= feature_div_value:
                    treenode = treenode.left
                else:
                    treenode = treenode.right
            if not isinstance(treenode.data, int):
                exit("some leafnode has not result")
            predicted_result_index = treenode.data
            print('result index is ' + str(predicted_result_index))
            results.append(self.leaf_labels[predicted_result_index])
        return results


# Backpropagation for Multi-Label Learning
class BPMLL:
    def __init__(self, epoch=20, regulization=0.0001, normalize=False):
        self.features = 0
        self.classes = 0
        self.samples = 0
        self.neural_num = 0
        self.normalize = normalize
        self.epoch = epoch
        self.neural_percent = 0.2
        self.learn_rate = 0.05
        # these two attributes affects the output
        self.weightsDecayCost = regulization
        self.error_small_change = 0.00001
        self.final_error = 0
        self.dataset = []
        self.threshold = None
        self.wsj = []
        self.vhs = []
        self.bias_b = []
        self.bias_a = []

    def fit(self, X, y):

        if isinstance(X, scipy.sparse.spmatrix):
            X_array = X.toarray()
        else:
            X_array = X

        self.samples = len(X_array)
        self.features = len(X_array[0])
        self.classes = len(y[0])

        self.dataset = self.prepare_data(X_array, y)
        self.samples = len(self.dataset)

        if self.samples < self.features:
            raise Exception("Your must have more instances than features")

        self.neural_num = round(self.features * self.neural_percent)

        self.wsj = [[(random.random() - 0.5) for j in range(self.classes)] for s in range(self.neural_num)]
        self.vhs = [[(random.random() - 0.5) for s in range(self.neural_num)] for h in range(self.features)]

        self.bias_b = [1 for i in range(self.classes)]
        self.bias_a = [1 for i in range(self.neural_num)]

        self.iterate_training()
        return self

    def prepare_data(self, X_array, y):
        dataset = []

        if self.normalize is True:
            X_array = BPMLL_models.Nomalizer(X_array, -0.8, 0.8).normalize()

        for i in range(self.samples):
            if np.sum(y) != 0 and np.sum(y) != self.classes:
                dataset.append(BPMLL_models.TrainPair(X_array[i], y[i]))

        return dataset

    def iterate_training(self):
        prev_error = self.global_error()
        # count = 0
        for ep in range(self.epoch):
            random.shuffle(self.dataset)
            for i in range(self.samples):
                self.fit_once(i)

            error = self.global_error()
            diff = prev_error - error
            if diff <= self.error_small_change * prev_error:
                # print('prev ' + str(prev_error))
                # print('now ' + str(error))
                # print('count is ' + str(count) )
                self.build_threshhold()
                self.final_error = error
                return
            prev_error = error
            # count += 1

        self.build_threshhold()
        self.final_error = prev_error
        return

    def fit_once(self, index):
        # skip samples whose Yi or n-Yi is an empty set
        x = self.dataset[index].attributes
        y = self.dataset[index].labels

        b, c = self.forward_propagation(x)

        d = [0 for j in range(self.classes)]

        isLabel = self.dataset[index].isLabel
        notLabel = self.dataset[index].notLabel
        isLabel_length = len(isLabel)
        notLabel_length = len(notLabel)

        for j in range(self.classes):
            dj_sigma = 0
            if y[j] == 1:
                for l in notLabel:
                    dj_sigma += math.exp(-(c[j] - c[l]))
            elif y[j] == 0:
                for k in isLabel:
                    dj_sigma -= math.exp(-(c[k] - c[j]))
            d[j] = (1 / (isLabel_length * notLabel_length)) * dj_sigma * (1 + c[j]) * (1 - c[j])

        e = [0 for s in range(self.neural_num)]

        for s in range(self.neural_num):
            es_sigma = 0
            for j in range(self.classes):
                es_sigma += d[j] * self.wsj[s][j]
            e[s] = es_sigma * (1 + b[s]) * (1 - b[s])

        for s in range(self.neural_num):
            for j in range(self.classes):
                self.wsj[s][j] += self.learn_rate * d[j] * b[s] - self.weightsDecayCost * self.wsj[s][j]

        for h in range(self.features):
            for s in range(self.neural_num):
                self.vhs[h][s] += self.learn_rate * e[s] * x[h] - self.weightsDecayCost * self.vhs[h][s]

        for j in range(self.classes):
            self.bias_b[j] += self.learn_rate * d[j] - self.weightsDecayCost * self.bias_b[j]

        for s in range(self.neural_num):
            self.bias_a[s] += self.learn_rate * e[s] - self.weightsDecayCost * self.bias_a[s]

        return

    def forward_propagation(self, x):
        netb = [0 for s in range(self.neural_num)]
        b = [0 for s in range(self.neural_num)]
        for s in range(self.neural_num):
            for h in range(self.features):
                netb[s] += self.vhs[h][s] * x[h]
            netb[s] += self.bias_a[s]
            b[s] = BPMLL_models.ActivationFunction().activate(netb[s])

        netc = [0 for j in range(self.classes)]
        c = [0 for j in range(self.classes)]
        for j in range(self.classes):
            for s in range(self.neural_num):
                netc[j] += self.wsj[s][j] * b[s]
            netc[j] += self.bias_b[j]
            c[j] = BPMLL_models.ActivationFunction().activate(netc[j])

        return b, c

    def global_error(self):
        global_error = 0
        weights_square_sum = 0

        for error_list in self.wsj:
            for error in error_list:
                weights_square_sum += error * error

        for error_list in self.vhs:
            for error in error_list:
                weights_square_sum += error * error

        for error in self.bias_b:
            weights_square_sum += error * error

        for error in self.bias_a:
            weights_square_sum += error * error

        for i in range(self.samples):
            sample_error_sigma = 0
            c = self.forward_propagation(self.dataset[i].attributes)[1]

            yi = self.dataset[i].isLabel
            nyi = self.dataset[i].notLabel
            yi_length = len(yi)
            nyi_length = len(nyi)

            # if yi_length != 0 and nyi_length != 0:
            for k in yi:
                for l in nyi:
                    sample_error_sigma += math.exp(-(c[k] - c[l]))
            global_error += 1 / (yi_length * nyi_length) * sample_error_sigma

        global_error += self.weightsDecayCost * 0.5 * weights_square_sum
        return global_error

    def build_threshhold(self):
        modelOutputs = []
        idealLabels = []
        for i in range(self.samples):
            c = self.forward_propagation(self.dataset[i].attributes)[1]
            modelOutputs.append(c)
            idealLabels.append(self.dataset[i].labels)

        self.threshold = BPMLL_models.ThresholdFunction(modelOutputs, idealLabels)

    def predict(self, X):
        samples, features = X.shape
        if features != self.features:
            raise Exception("inconsistent feature dimension")

        X_array = X.toarray()
        if self.normalize is True:
            X_array = BPMLL_models.Nomalizer(X_array, -0.8, 0.8).normalize()

        result = BPMLL_models.BPMLLResults(self.final_error)
        for i in range(samples):
            sample_result = []
            topLabel = None
            c = self.forward_propagation(X_array[i])[1]
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

