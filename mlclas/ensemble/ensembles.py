"""
This file implements some ensemble classifiers.
Some algorithms may not work well beacause they are implemented during my early work,
including CalibratedLabelRanking and RandomKLabelsets.
"""

import numpy as np
import copy
import math
import random
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.neighbors import NearestNeighbors


class BinaryRelevance:
    # A simple wrapper for OneVsRestClassifier
    def __init__(self, estimator):
        self.classifier = OneVsRestClassifier(estimator)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        binary_result = self.classifier.predict(X)
        sample_num, classes = binary_result.shape
        y_reverse = []
        for i in range(sample_num):
            sample_label = [j for j in range(classes) if binary_result[i][j] == 1]
            y_reverse.append(sample_label)
        return y_reverse


class ClassifierChains:
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators = []
        self.classes = 0

    def fit(self, X, y):
        self.estimators = []
        self.classes = y.shape[1]
        for i in range(self.classes):
            temp_column = y[:, i]
            temp_estimator = copy.deepcopy(self.estimator).fit(X, temp_column)
            self.estimators.append(temp_estimator)
            temp_column = np.array([temp_column.tolist()])
            temp_column = csr_matrix(temp_column.T)
            X = hstack([X, temp_column])
        return self

    def predict(self, X):
        result = []
        dataset_length = X.shape[0]
        class_num = len(self.estimators)

        # Initialize result array
        for i in range(dataset_length):
            result.append([])

        for j in range(class_num):
            temp_result = self.estimators[j].predict(X)

            for k in range(dataset_length):
                if temp_result[k] == 1.0:
                    result[k].append(j)

            temp_result = np.array([temp_result]).T.tolist()
            temp_result = csr_matrix(temp_result)

            X = hstack([X, temp_result])

        return result


class CalibratedLabelRanking:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimators_ = []
        self.samples = y.shape[0]
        self.classes_ = y.shape[1]
        self.virtual_label = self.classes_

        for i in range(0, self.classes_):
            for j in range(i + 1, self.classes_):
                temp_estimator = copy.deepcopy(self.estimator)
                data = None
                target = []
                y_i = y[:, i]
                y_j = y[:, j]
                for index in range(self.samples):
                    if y_i[index] + y_j[index] == 1:
                        data = vstack([data, X.getrow(index)]) if data is not None else X.getrow(index)
                        target.append(i if y_i[index] == 1 else j)

                if target.count(target[0]) == len(target) or len(target) == 0:
                    continue
                self.estimators_.append(temp_estimator.fit(data, target))

        for i in range(0, self.classes_):
            target = []
            temp_estimator = copy.deepcopy(self.estimator)
            y_i = y[:, i]
            for index in range(self.samples):
                target.append(i if y_i[index] == 1 else self.virtual_label)

            if target.count(target[0]) == len(target) or len(target) == 0:
                continue
            self.estimators_.append(temp_estimator.fit(X, target))
        return self

    def predict(self, X):
        test_samples = X.shape[0]
        count = [{} for i in range(test_samples)]
        for estimator_ in self.estimators_:
            res = estimator_.predict(X)
            for i in range(test_samples):
                tmp = res[i]
                if tmp in count[i]:
                    count[i][tmp] += 1
                else:
                    count[i][tmp] = 1

        result = []
        for single_count in count:
            one_res = []
            threshold = single_count[self.virtual_label] if self.virtual_label in single_count else 0
            for entry in single_count:
                if single_count[entry] > threshold:
                    one_res.append(entry)
            result.append(sorted(one_res))
        return result


class RandomKLabelsets:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = y.shape[1]
        self.samples = y.shape[0]
        self.maps = []
        self.estimators_ = []
        # TODO:k should be changed
        k = 3
        n = 2 * self.classes_
        labels = [i for i in range(self.classes_)]
        self.k_labelsets = []
        for i in range(n):
            labelset = sorted(random.sample(labels, k))
            while labelset in self.k_labelsets:
                labelset = sorted(random.sample(labels, k))
            self.k_labelsets.append(labelset)

        # Reverse Multilabel Minarization
        y_reverse = []
        for i in range(self.samples):
            sample_label = [j for j in range(self.classes_) if y[i][j] == 1]
            y_reverse.append(sample_label)

        for each_set in self.k_labelsets:
            temp_estimator = copy.deepcopy(self.estimator)
            class_map = []
            data = None
            target = []

            for index in range(self.samples):
                intersection = [x for x in y_reverse[index] if x in each_set]
                if len(intersection) == 0:
                    continue
                if intersection not in class_map:
                    class_map.append(intersection)
                target.append(class_map.index(intersection))
                data = vstack([data, X.getrow(index)]) if data is not None else X.getrow(index)

            self.maps.append(class_map)
            self.estimators_.append(temp_estimator.fit(data, target))
        return self

    def predict(self, X):
        test_samples = X.shape[0]
        result = [{} for i in range(test_samples)]

        max_votes = [0 for i in range(self.classes_)]
        for labelset in self.k_labelsets:
            for index in labelset:
                max_votes[index] += 1

        for estimator_id in range(len(self.estimators_)):
            estimator = self.estimators_[estimator_id]
            res = estimator.predict(X)
            for index in range(test_samples):
                actual_res = self.maps[estimator_id][res[index]]
                for label in actual_res:
                    if label in result[index]:
                        result[index][label] += 1
                    else:
                        result[index][label] = 1

        return_result = []
        for each_result in result:
            label_result = []
            for i in each_result:
                if each_result[i] / max_votes[i] > 0.5:
                    label_result.append(i)
            return_result.append(label_result)

        return return_result


class MLKNN:
    def __init__(self, k):
        self.knn = None
        self.samples = 0
        self.classes = 0
        self.k = k
        self.y = None
        self.s = 1
        self.ph = None
        self.kj = None
        self.knj = None

    def fit(self, X, y):
        X = csr_matrix(X)
        self.samples = y.shape[0]
        self.classes = y.shape[1]
        self.knn = NearestNeighbors(self.k)
        self.knn.fit(X)

        self.ph = np.sum(y, axis=0)
        # Reverse multilabel minarization and prepare for P(Hj)
        y_reverse = []
        for i in range(self.samples):
            sample_label = [j for j in range(self.classes) if y[i][j] == 1]
            for label in sample_label:
                self.ph[label] += 1
            y_reverse.append(sample_label)
        self.y = y_reverse

        self.ph = (self.s + self.ph) / (2 * self.s + self.samples)
        self.ph /= 1 - self.ph

        self.kj = np.zeros((self.classes, self.k + 1))
        self.knj = np.zeros((self.classes, self.k + 1))

        for index in range(self.samples):
            sample_label = y_reverse[index]
            neighbors = self.knn.kneighbors(X.getrow(index), n_neighbors=self.k + 1, return_distance=False)[0][1:]
            neighbor_label_count = [0 for i in range(self.classes)]
            for neighbor in neighbors:
                neighbor_label = y_reverse[neighbor]
                for each_label in neighbor_label:
                    neighbor_label_count[each_label] += 1

            for label_index in range(self.classes):
                if label_index in sample_label:
                    self.kj[label_index][neighbor_label_count[label_index]] += 1
                else:
                    self.knj[label_index][neighbor_label_count[label_index]] += 1

        return self

    def predict(self, X):
        X = csr_matrix(X)
        test_samples_length = X.shape[0]
        res = [[] for i in range(test_samples_length)]

        for index in range(test_samples_length):
            neighbors = self.knn.kneighbors(X.getrow(index), n_neighbors=self.k + 1, return_distance=False)[0][1:]
            neighbor_label_count = [0 for i in range(self.classes)]
            for neighbor in neighbors:
                neighbor_label = self.y[neighbor]
                for each_label in neighbor_label:
                    neighbor_label_count[each_label] += 1

            for each_label in range(self.classes):
                pch = (self.s + self.kj[each_label][neighbor_label_count[each_label]]) / (
                    self.s * (self.k + 1) + sum(self.kj[each_label]))
                pcnh = (self.s + self.knj[each_label][neighbor_label_count[each_label]]) / (
                    self.s * (self.k + 1) + sum(self.knj[each_label]))
                probability = self.ph[each_label] * pch / pcnh
                if probability > 1:
                    res[index].append(each_label)

        return res
