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

    def fit(self, X, y):
        self.estimators_ = []
        self.classes_ = y.shape[1]
        for i in range(self.classes_):
            temp_column = y[:, i]
            temp_estimator = copy.deepcopy(self.estimator).fit(X, temp_column)
            self.estimators_.append(temp_estimator)
            temp_column = np.array([temp_column.tolist()])
            temp_column = csr_matrix(temp_column.T)
            X = hstack([X, temp_column])
        return self

    def predict(self, X):
        result = []
        dataset_length = X.shape[0]
        class_num = len(self.estimators_)

        # Initialize result array
        for i in range(dataset_length):
            result.append([])

        for j in range(class_num):
            temp_result = self.estimators_[j].predict(X)

            for k in range(dataset_length):
                if temp_result[k] == 1.0:
                    result[k].append()

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
    def __init__(self):
        self.knn = None
        self.samples = 0
        self.classes = 0
        self.k = 0
        self.y = None

    def fit(self, X, y):
        self.s = 1
        self.samples = y.shape[0]
        self.classes = y.shape[1]
        self.k = round(math.sqrt(self.samples))
        self.knn = NearestNeighbors(self.k)
        self.knn.fit(X)

        self.ph = [0 for i in range(self.classes)]
        # Reverse multilabel minarization and prepare for P(Hj)
        y_reverse = []
        for i in range(self.samples):
            sample_label = [j for j in range(self.classes) if y[i][j] == 1]
            for label in sample_label:
                self.ph[label] += 1
            y_reverse.append(sample_label)

        self.y = y_reverse
        for i in range(self.classes):
            self.ph[i] = (self.s + self.ph[i]) / (2 * self.s + self.samples)
            self.ph[i] /= (1 - self.ph[i])

        self.kj = [[0 for j in range(self.k + 1)] for i in range(self.classes)]
        self.knj = [[0 for j in range(self.k + 1)] for i in range(self.classes)]

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


class TreeNode(object):
    def __init__(self, data=None):
        self.left = None
        self.right = None
        self.data = data


class MultiLabelDecisionTree:
    def __init__(self):
        self.samples = 0
        self.classes = 0
        self.y_reverse = []
        self.stop_criterion = 0
        self.root = None
        self.leaf_labels = []
        self.leaf_index = 0
        self.features = 0

    def fit(self, X, y):
        self.features = X.shape[1]
        self.samples, self.classes = y.shape
        self.splitting_num = int(math.sqrt(self.samples))
        self.leaf_labels = []
        self.leaf_index = 0
        if self.samples < self.splitting_num:
            raise Exception("Too few samples")
        self.root = TreeNode()
        self.fit_tree(X, y, self.root)
        print(self.leaf_labels)
        return self

    def fit_tree(self, X, y, tree_node, mlent_all=0):
        if mlent_all == 0:
            mlent_all = self.get_mlent(y)

        sample_num = X.shape[0]
        if sample_num <= self.splitting_num:
            labels = []
            for i in range(self.classes):
                if np.sum(y[:, i]) / sample_num > 0.5:
                    labels.append(i)

            tree_node.data = self.leaf_index
            self.leaf_labels.append(labels)
            self.leaf_index += 1
            return

        splitting_point = [sample_num * x // self.splitting_num for x in range(1, self.splitting_num)]
        all_statistics = []

        for feature_index in range(self.features):
            feature_values = X.getcol(feature_index).toarray()
            sorted_feature_values = sorted(feature_values)
            for point in splitting_point:
                y_left = []
                y_right = []
                spp = (sorted_feature_values[point] + sorted_feature_values[point + 1]) / 2
                for sample_index in range(sample_num):
                    if feature_values[sample_index] <= spp:
                        y_left.append(y[sample_index].tolist())
                    else:
                        y_right.append(y[sample_index].tolist())
                y_left = np.array(y_left)
                y_right = np.array(y_right)
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    all_statistics.append(0)
                else:
                    mlent_minus = self.get_mlent(y_left)
                    mlent_plus = self.get_mlent(y_right)
                    ig = mlent_all - (
                        y_left.shape[0] / sample_num * mlent_minus + y_right.shape[0] / sample_num * mlent_plus)
                    all_statistics.append(ig)

        max_index = 0
        max_ig = 0
        for ig_index in range(len(all_statistics)):
            if all_statistics[ig_index] > max_ig:
                max_ig = all_statistics[ig_index]
                max_index = ig_index

        feature_index, remainder = divmod(max_index, self.splitting_num - 1)
        feature_values = X.getcol(feature_index).toarray()
        sorted_feature_values = sorted(feature_values)
        split_value = (sorted_feature_values[splitting_point[remainder]] + sorted_feature_values[
            splitting_point[remainder] + 1]) / 2

        tree_node.data = [feature_index, split_value]
        X_left = None
        y_left = []
        X_right = None
        y_right = []
        for sample_index in range(sample_num):
            if feature_values[sample_index] <= split_value:
                X_left = vstack([X_left, X.getrow(sample_index)]) if X_left is not None else X.getrow(sample_index)
                y_left.append(y[sample_index].tolist())
            else:
                X_right = vstack([X_right, X.getrow(sample_index)]) if X_right is not None else X.getrow(sample_index)
                y_right.append(y[sample_index].tolist())

        if len(y_left) != 0:
            tree_node.left = TreeNode()
            self.fit_tree(X_left, np.array(y_left), tree_node.left)
        if len(y_right) != 0:
            tree_node.right = TreeNode()
            self.fit_tree(X_right, np.array(y_right), tree_node.right)

    def get_mlent(self, y):
        samples, classes = y.shape
        pj = [np.sum(y[:, i]) / samples for i in range(classes)]
        sum = 0
        for j in range(classes):
            pjs = pj[j]
            if pjs == 0 or pjs == 1:
                continue

            sum += -pjs * math.log(pjs, 2) - (1 - pjs) * math.log((1 - pjs), 2)

        return sum

    def predict(self, X):
        predict_num, predict_features = X.shape
        if predict_features != self.features:
            exit("inconsistent feature number")

        results = []
        for i in range(predict_num):
            feature_values = X.getrow(i).toarray()[0]
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
            results.append(self.leaf_labels[predicted_result_index])
        return results


