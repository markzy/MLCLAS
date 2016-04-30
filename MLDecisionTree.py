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
        self.useStandardError = True

    def fit(self, X, y, minNum=5):
        instances = DTModels.MLInstaces(X, y)
        self.features = instances.features
        self.classes = instances.classes
        self.minNum = minNum

        self.fit_tree(instances, self.root)
        self.root.prune()
        self.learned = True
        return self

    # to-do: construct leaf node;check how to avoid less than 2
    def fit_tree(self, data, treenode):
        if data.samples <= 2 * self.minNum or data.pure is True:
            treenode.leaf(data)
            return

        select_model = DTModels.ModelSelection(useMDL=True, minNum=self.minNum)
        attr, value = select_model.select(data)
        left_data, right_data, distribution = data.split(attr, value)

        treenode.distribution = distribution
        treenode.splitInfo = [attr, value]
        treenode.left = DTModels.TreeNode()
        treenode.right = DTModels.TreeNode()

        # split data
        self.fit_tree(left_data, treenode.left)
        self.fit_tree(right_data, treenode.right)

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
            while not treenode.isLeaf:
                attrIndex, splitValue = treenode.splitInfo
                if values[attrIndex] <= splitValue:
                    treenode = treenode.left
                else:
                    treenode = treenode.right
            results.append(treenode.getPrediectedLabels())

        return results
