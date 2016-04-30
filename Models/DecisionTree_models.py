import numpy as np
import math
import scipy.sparse
import operator
import sys


# Definition of a simple tree node
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.distribution = None

        self.isLeaf = False
        self.splitInfo = None

    def leaf(self, instances):
        self.isLeaf = True
        if instances.samples == 0:
            return

        self.distribution = Distribution(instances)

    def getEstimatedError(self, useStandardError=True):
        if self.isLeaf:
            return self.distribution.numIncorrect()
        elif useStandardError:
            totalnum = self.distribution.total() * self.distribution.classes
            errors = self.left.getEstimatedError() + self.right.getEstimatedError()
            standard_error = math.sqrt(errors * (totalnum - errors) / totalnum)
            return errors + standard_error
        else:
            return self.left.getEstimatedError() + self.right.getEstimatedError()

    def getPrediectedLabels(self):
        return self.distribution.predictedLabels()

    def prune(self):
        if self.isLeaf is True:
            return

        self.left.prune()
        self.right.prune()

        error_node = self.distribution.numIncorrect()
        error_tree = self.getEstimatedError()

        # prune the subtree if true
        if error_node <= error_tree:
            self.left = None
            self.right = None
            self.isLeaf = True
            self.splitInfo = None

        return


class MLInstaces:
    def __init__(self, X, y):
        self.samples = 0
        self.features = 0
        self.classes = 0
        self.pure = False
        self.all_attributes = None
        self.bin_labels = None
        self.instances = np.array([])
        self.initialize(X, y)

    def initialize(self, X, y):
        if isinstance(X, scipy.sparse.spmatrix):
            X_array = X.toarray()
        else:
            X_array = np.array(X)

        y = np.array(y)
        self.samples, self.features = X_array.shape
        self.classes = y.shape[1]
        self.all_attributes = X_array
        self.bin_labels = y
        self.instances = np.array([i for i in range(self.samples)])

        # pure test
        y_list = y.tolist()
        self.pure = y_list.count(y_list[0]) == len(y_list)

    def sort(self, attrIndex):
        attr_values = self.all_attributes[:, attrIndex]

        attr_dic = {}
        for i in range(self.samples):
            attr_dic[i] = attr_values[i]

        sorted_output = sorted(attr_dic.items(), key=operator.itemgetter(1))
        self.instances = np.array([sorted_tuple[0] for sorted_tuple in sorted_output])
        return True

    def split(self, attrIndex, splitValue):
        attrValues = self.all_attributes[:, attrIndex]
        left = []
        right = []
        for index in range(self.samples):
            if attrValues[index] <= splitValue:
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
    def __init__(self, data):
        if not isinstance(data, tuple):
            self.classes = data.classes
            self.left = np.zeros(data.classes)
            self.right = np.sum(data.bin_labels[data.instances], axis=0)
            self.leftNum = 0
            self.rightNum = data.samples
        else:
            self.left = np.sum(data[0], axis=0)
            self.right = np.sum(data[1], axis=0)
            self.leftNum = len(data[0])
            self.rightNum = len(data[1])
            self.classes = len(self.left)

    def shiftLeft(self, from_, to_, data):
        pop_up_indices = data.instances[from_:to_]
        shift_weight = np.sum(data.bin_labels[pop_up_indices], axis=0)
        shift_num = to_ - from_
        self.leftNum += shift_num
        self.rightNum -= shift_num
        self.right = self.right - shift_weight
        self.left = self.left + shift_weight

    def total(self):
        return self.leftNum + self.rightNum

    def perClass(self):
        return self.left + self.right

    def numIncorrect(self):
        perClassTotal = self.perClass()
        predicted = self.predictedLabels()
        numTotal = self.total()
        sum_temp = np.zeros(self.classes)
        for index in predicted:
            sum_temp[index] = numTotal
        return np.sum(np.fabs(perClassTotal - sum_temp))

    def predictedLabels(self):
        perClassTotal = self.perClass()
        numTotal = self.total()
        labels = []
        count = 0
        for index in range(len(perClassTotal)):
            if perClassTotal[index] > numTotal / 2:
                count += 1
                labels.append(index)
        if count == 0:
            max_index, max_value = max(enumerate(perClassTotal), key=operator.itemgetter(1))
            labels.append(max_index)

        return labels


# only support numeric attributes now
class ModelSelection:
    def __init__(self, useMDL=False, minNum=2):
        self.useMDL = useMDL
        self.minNum = minNum

    def select(self, instances):
        bestAttr = 0
        bestSplitValue = 0
        currentInfo = sys.float_info.max
        for i in range(instances.features):
            model = C45Split(i, self.useMDL, self.minNum)
            result = model.build(instances)
            if result is None:
                continue
            else:
                splitValue, info = result
            if info < currentInfo:
                bestAttr = i
                bestSplitValue = splitValue
                currentInfo = info

        return bestAttr, bestSplitValue


class C45Split:
    def __init__(self, attrIndex, useMDL, minNum=2):
        self.attrIndex = attrIndex
        self.useMDL = useMDL
        self.minor = 1e-5
        self.minNum = 2

    def build(self, data):
        last = 0
        count = 0
        splitIndex = 0

        minNum = int(0.1 * data.samples / data.classes)
        if minNum <= self.minNum:
            minNum = self.minNum
        elif minNum > 25:
            minNum = 25

        attrIndex = self.attrIndex
        data.sort(self.attrIndex)
        distribution = Distribution(data)
        currentInfo = Entropy.getEntropy(distribution)
        attrValues = data.all_attributes[:, attrIndex][data.instances]
        for i in range(1, data.samples):
            # equals to ignore bits
            if (attrValues[i - 1] + self.minor) < attrValues[i]:
                distribution.shiftLeft(last, i, data)
                last = i
                if distribution.leftNum > minNum and distribution.rightNum > minNum:
                    count += 1
                    newInfo = Entropy.getEntropy(distribution)
                    if newInfo < currentInfo:
                        currentInfo = newInfo
                        splitIndex = i

        if count == 0:
            return

        bestSplitValue = (attrValues[splitIndex - 1] + attrValues[splitIndex]) / 2

        if bestSplitValue == attrValues[splitIndex]:
            bestSplitValue = attrValues[splitIndex - 1]

        if self.useMDL:
            currentInfo += math.log2(count) / data.samples

        return bestSplitValue, currentInfo


class Entropy:
    @staticmethod
    def getMLEnt(y, samples):
        if samples < 1:
            return 0
        mlent = 0
        pi = y / samples
        for prob in pi:
            if prob != 0 and prob != 1:
                mlent += -prob * math.log2(prob) - (1 - prob) * math.log2((1 - prob))
        return mlent

    @staticmethod
    def getEntropy(distribution):
        left = distribution.left
        right = distribution.right
        leftNum = distribution.leftNum
        rightNum = distribution.rightNum
        entropy = (leftNum * Entropy.getMLEnt(left, leftNum) + rightNum * Entropy.getMLEnt(right, rightNum)) / (leftNum + rightNum)

        return entropy
