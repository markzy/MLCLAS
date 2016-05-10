import numpy as np


# models designed for RankingSVM
class AllLabelInfo:
    def __init__(self):
        # range and sum information
        self.totalProduct = 0
        self.eachRange = []
        # label information
        self.labels = []
        self.notLabels = []
        self.labelsNum = []
        self.notLabelsNum = []

    def append(self, label_array, not_array):
        self.labels.append(label_array)
        self.notLabels.append(not_array)
        self.labelsNum.append(len(label_array))
        self.notLabelsNum.append(len(not_array))

        """ update range information
            while this range information corresponds to Python customs, i.e. range(a,b) means [a,b) in math
        """
        newIndex = self.totalProduct
        product = len(label_array) * len(not_array)
        self.eachRange.append((newIndex, newIndex + product))

        self.totalProduct += product

    def getShape(self, index, elaborate=False):
        if elaborate is False:
            return self.labelsNum[index], self.notLabelsNum[index]
        else:
            return (self.labelsNum[index], self.notLabelsNum[index]), self.labels[index], self.notLabels[index]

    def ravel_multiple_index(self):
        pass

    def getRangeFromIndex(self, index):
        return self.eachRange[index]
