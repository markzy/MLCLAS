import numpy as np
import scipy.sparse
from Models import DecisionTree_models as DTModels


# to-do: stop criterion can be refined
class MLDecisionTree:
    def __init__(self):
        self.features = 0
        self.classes = 0
        self.root = DTModels.TreeNode()
        self.stop_criterion = 0
        self.learned = False
        self.minNum = 2

    def fit(self, X, y, minNum=2):
        instances = DTModels.MLInstaces(X, y)
        self.features = instances.features
        self.classes = instances.classes
        self.minNum = minNum

        self.fit_tree(instances, self.root)
        self.learned = True
        return self

    # to-do: construct leaf node;check how to avoid less than 2
    def fit_tree(self, data, treenode):
        if data.samples <= 2 * self.minNum or data.pure is True:
            treenode.leaf(data)
            # treenode.leaf = True
            # if data.samples == 0:
            #     return
            #

            return

        select_model = DTModels.ModelSelection(useMDL=True,minNum=self.minNum)
        attr, value, distribution = select_model.select(data)
        treenode.distribution = distribution
        treenode.splitInfo = [attr, value]
        treenode.left = DTModels.TreeNode()
        treenode.right = DTModels.TreeNode()

        # split data
        left_data, right_data = data.split(attr, value)
        self.fit_tree(left_data, treenode.left)
        self.fit_tree(right_data, treenode.right)

    def buildLeaf(self, data):

    def prune(self, treenode):

    def predict(self, X):
        if self.learned is False:
            raise Exception('this tree has not been fitted')

        if isinstance(X, scipy.sparse.spmatrix):
            X_array = X.toarray()
        else:
            X_array = np.array(X)

        samples, features = X.shape
        if features != self.features:
            raise Exception('inconsistent attribute number')

        results = []
        for index in range(samples):
            values = X_array[index]
            treenode = self.root
            while treenode.leaf is False:
                attrIndex, splitValue = treenode.splitInfo
                if values[attrIndex] <= splitValue:
                    treenode = treenode.left
                else:
                    treenode = treenode.right
            results.append(treenode.predictedLabels)

        return results
