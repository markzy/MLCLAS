import numpy as np
import math
import operator
from mlclas.utils import check_feature_input, check_target_input


class Countobj:
    def __init__(self):
        self.a = 0


# Definition of a simple tree node
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.distribution = None

        self.is_leaf = False
        self.split_info = None

    def leaf(self, instances):
        """
        function that makes this node a leaf
        :param instances: MLInstances object
        :return: None
        """
        self.is_leaf = True
        if instances.samples == 0:
            return

        self.distribution = Distribution(instances)

    def get_estimated_errors(self):
        if self.is_leaf:
            return TreeNode.get_distribution_error(self.distribution)
        else:
            return self.left.get_estimated_errors() + self.right.get_estimated_errors()

    @staticmethod
    def get_distribution_error(distribution):
        if distribution is not None:
            errors = distribution.num_incorrect()
            regular = 1 / 2
            return errors + regular

    def get_subtree_errors(self):
        if self.is_leaf:
            return TreeNode.get_distribution_error(self.distribution)

        totalnum = self.distribution.total() * self.distribution.classes
        errors = self.get_estimated_errors()
        standard_error = math.sqrt(errors * (totalnum - errors) / totalnum)
        return errors + standard_error

    def get_prediected_labels(self):
        return self.distribution.predicted_labels()

    def access_by_index(self, index):
        if index == 0:
            return self.left
        return self.right

    def prune(self, raise_subtree, countobj):
        """
        Use pessimistic pruning strategy desingned by myself.
        It depends on the error calculation of the nodes and subtree.
        """
        if self.is_leaf is True:
            return

        self.left.prune(raise_subtree, countobj)
        self.right.prune(raise_subtree, countobj)

        # error as a node
        error_node = TreeNode.get_distribution_error(self.distribution)
        # error as a tree
        error_tree = self.get_subtree_errors()

        # error for the largest branch
        if raise_subtree:
            if self.left.distribution.total() >= self.right.distribution.total():
                branch_index = 0
                error_largest_brach = self.left.get_subtree_errors()
            else:
                branch_index = 1
                error_largest_brach = self.right.get_subtree_errors()
        else:
            error_largest_brach = float('inf')

        # turn the subtree into a single node if true
        if error_node <= error_tree and error_node <= error_largest_brach:
            countobj.a += 1
            self.left = None
            self.right = None
            self.is_leaf = True
            self.split_info = None

        # raise the subtree if true, not possible when raise_subtree is False
        if error_largest_brach < error_tree:
            node = self.access_by_index(branch_index)
            self.distribution = node.distribution
            self.left = node.left
            self.right = node.right
            self.is_leaf = node.is_leaf
            self.split_info = node.split_info

        return


class MLInstaces:
    """
    Definition of a datasets structure that wraps all the training examples and provide useful methods.
    """

    def __init__(self, x, y):
        self.samples = 0
        self.features = 0
        self.classes = 0
        self.pure = False
        self.all_attributes = None
        self.bin_labels = None
        self.instances = np.array([])
        self.initialize(x, y)

    def initialize(self, x, y):
        x = check_feature_input(x)
        y = check_target_input(y)

        self.samples, self.features = x.shape
        self.classes = y.shape[1]
        self.all_attributes = x
        self.bin_labels = y
        self.instances = np.array([i for i in range(self.samples)])

        # pure tests: check if all the samples belong to same label set
        y_list = y.tolist()
        self.pure = y_list.count(y_list[0]) == len(y_list)

    def sort(self, attr_index):
        """
        method that sort the training samples by index, according to a specific attribute
        """
        attr_values = self.all_attributes[:, attr_index]

        attr_dic = {}
        for i in range(self.samples):
            attr_dic[i] = attr_values[i]

        sorted_output = sorted(attr_dic.items(), key=operator.itemgetter(1))
        self.instances = np.array([sorted_tuple[0] for sorted_tuple in sorted_output])
        return True

    def split(self, attr_index, split_value):
        """
        returns two split MLInstances according to the attr_index and split_value
        """
        attr_values = self.all_attributes[:, attr_index]
        left = []
        right = []
        for index in range(self.samples):
            if attr_values[index] <= split_value:
                left.append(index)
            else:
                right.append(index)

        left_indices = np.array(left)
        right_indices = np.array(right)
        left_labels = self.bin_labels[left_indices]
        right_labels = self.bin_labels[right_indices]
        return_distribution = Distribution((left_labels, right_labels))

        left_instances = MLInstaces(self.all_attributes[left_indices], left_labels)
        right_instances = MLInstaces(self.all_attributes[right_indices], right_labels)
        return left_instances, right_instances, return_distribution


class Distribution:
    """
    fundamental class!
    It records the distribution of samples which corresponds to the distribution of a binary tree!
    """

    def __init__(self, data):
        if not isinstance(data, tuple):
            # initializing via MLInstances
            self.classes = data.classes
            self.left = np.zeros(data.classes)
            self.right = np.sum(data.bin_labels[data.instances], axis=0)
            self.left_num = 0
            self.right_num = data.samples
        else:
            # initializing via (left_labels, right_labels)
            self.left = np.sum(data[0], axis=0)
            self.right = np.sum(data[1], axis=0)
            self.left_num = len(data[0])
            self.right_num = len(data[1])
            self.classes = len(self.left)

    def shift_left(self, from_, to_, data):
        pop_up_indices = data.instances[from_:to_]
        shift_weight = np.sum(data.bin_labels[pop_up_indices], axis=0)
        shift_num = to_ - from_
        self.left_num += shift_num
        self.right_num -= shift_num
        self.right = self.right - shift_weight
        self.left = self.left + shift_weight

    def total(self):
        return self.left_num + self.right_num

    def per_class(self):
        return self.left + self.right

    def num_incorrect(self):
        # return the number of incorrect predictions
        perclass_total = self.per_class()
        predicted = self.predicted_labels()
        num_total = self.total()
        sum_temp = np.zeros(self.classes)
        for index in predicted:
            sum_temp[index] = num_total
        return np.sum(np.fabs(perclass_total - sum_temp))

    def predicted_labels(self):
        # return the predicted label set according to this distribution
        perclass_total = self.per_class()
        num_total = self.total()
        labels = []
        count = 0
        for index in range(len(perclass_total)):
            if perclass_total[index] > num_total / 2:
                count += 1
                labels.append(index)
        if count == 0:
            max_index, max_value = max(enumerate(perclass_total), key=operator.itemgetter(1))
            labels.append(max_index)

        return labels


# only support numeric attributes now
class ModelSelection:
    """
    CLass that help select the best split feature and split value
    """

    def __init__(self, use_mdl=False, min_num=2):
        self.use_mdl = use_mdl
        self.min_num = min_num

    def select(self, instances):
        best_attr = None
        best_split_value = None
        current_info = float('inf')

        # check if there are enough instances
        min_num = int(0.1 * instances.samples / instances.classes)
        if min_num <= self.min_num:
            min_num = self.min_num
        elif min_num > 25:
            min_num = 25

        # min_num = 2
        if instances.samples < 2 * min_num:
            return

        for i in range(instances.features):
            model = C45Split(i, self.use_mdl, min_num)
            result = model.build(instances)
            if result is None:
                continue
            else:
                split_value, info = result
            """
            usually we calculate information gain, for which larger is better.
            To simplify, I calculate new_info in the formula [info_gain = old_info - new_info],
            for which smaller is better.
            """
            if info < current_info:
                best_attr = i
                best_split_value = split_value
                current_info = info

        if best_attr is None:
            return

        return best_attr, best_split_value


class C45Split:
    def __init__(self, attr_index, use_mdl, min_num=2):
        self.attr_index = attr_index
        self.use_mdl = use_mdl
        self.minor = 1e-5
        self.min_num = min_num

    def build(self, data):
        last = 0
        count = 0
        split_index = 0

        min_num = self.min_num
        attr_index = self.attr_index

        data.sort(self.attr_index)
        distribution = Distribution(data)

        current_info = Entropy.get_entropy(distribution)
        attr_values = data.all_attributes[:, attr_index][data.instances]

        for i in range(1, data.samples):
            # equals to ignore bits
            if (attr_values[i - 1] + self.minor) < attr_values[i]:
                distribution.shift_left(last, i, data)
                last = i
                if distribution.left_num > min_num and distribution.right_num > min_num:
                    count += 1
                    new_info = Entropy.get_entropy(distribution)
                    if new_info < current_info:
                        current_info = new_info
                        split_index = i

        if count == 0:
            return

        best_split_value = (attr_values[split_index - 1] + attr_values[split_index]) / 2

        if best_split_value == attr_values[split_index]:
            best_split_value = attr_values[split_index - 1]

        """
        MDL correction, which is introduced in:
        >   Quinlan, J. Ross. "Improved use of continuous attributes in C4. 5."
            Journal of artificial intelligence research (1996): 77-90.
        """
        if self.use_mdl:
            current_info += math.log2(count) / data.samples

        return best_split_value, current_info


class Entropy:
    """ class that helps calculate entrophy introduced in the paper"""

    @staticmethod
    def get_mlent(y, samples):
        if samples < 1:
            return 0
        mlent = 0
        pi = y / samples
        for prob in pi:
            if prob != 0 and prob != 1:
                mlent += -prob * math.log2(prob) - (1 - prob) * math.log2((1 - prob))
        return mlent

    @staticmethod
    def get_entropy(distribution):
        left = distribution.left
        right = distribution.right
        left_num = distribution.left_num
        right_num = distribution.right_num
        entropy = (left_num * Entropy.get_mlent(left, left_num) + right_num * Entropy.get_mlent(right, right_num)) / (left_num + right_num)

        return entropy
