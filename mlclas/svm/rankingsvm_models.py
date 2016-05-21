# models designed for RankingSVM
class AllLabelInfo:
    def __init__(self):
        # range and sum information
        self.eachProduct = []
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
        new_index = self.totalProduct
        product = len(label_array) * len(not_array)
        self.eachRange.append((new_index, new_index + product))

        self.eachProduct.append(product)
        self.totalProduct += product

    def get_shape(self, index, elaborate=False):
        if elaborate is False:
            return self.labelsNum[index], self.notLabelsNum[index]
        else:
            return (self.labelsNum[index], self.notLabelsNum[index]), self.labels[index], self.notLabels[index]

    def get_range(self, index):
        return self.eachRange[index]

    def get_each_product(self, index):
        return self.eachProduct[index]
