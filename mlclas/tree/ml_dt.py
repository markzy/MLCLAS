import numpy as np
import scipy.sparse
import mlclas.tree.dt_models as dtm


# to-do: stop criterion can be refined
class MLDecisionTree:
    def __init__(self):
        self.features = 0
        self.classes = 0
        self.root = dtm.TreeNode()
        self.stop_criterion = 0
        self.learned = False
        self.min_num = 2
        self.useStandardError = True

    def fit(self, X, y, min_num=5):
        instances = dtm.MLInstaces(X, y)
        self.features = instances.features
        self.classes = instances.classes
        self.min_num = min_num

        self.fit_tree(instances, self.root)
        self.root.prune()
        self.learned = True
        return self

    # to-do: construct leaf node;check how to avoid less than 2
    def fit_tree(self, data, treenode):
        if data.samples <= 2 * self.min_num or data.pure is True:
            treenode.leaf(data)
            return

        select_model = dtm.ModelSelection(use_mdl=True, min_num=self.min_num)
        attr, value = select_model.select(data)
        left_data, right_data, distribution = data.split(attr, value)

        treenode.distribution = distribution
        treenode.split_info = [attr, value]
        treenode.left = dtm.TreeNode()
        treenode.right = dtm.TreeNode()

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
            while not treenode.is_leaf:
                attr_index, split_value = treenode.split_info
                if values[attr_index] <= split_value:
                    treenode = treenode.left
                else:
                    treenode = treenode.right
            results.append(treenode.get_prediected_labels())

        return results
