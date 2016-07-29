from mlclas.utils import check_feature_input
from mlclas.tree import dt_models as dtm
from mlclas.stats import Normalizer


class MLDecisionTree:
    """
    Multilabel Decision Tree, the learning process is based on:
    >   Clare, Amanda, and Ross D. King. "Knowledge discovery in multi-label phenotype datasets."
        Principles of datasets mining and knowledge discovery. Springer Berlin Heidelberg, 2001. 42-53.
    The pruning strategy is derived from:
    >   Mingers, John. "An empirical comparison of pruning methods for decision tree induction."
        Machine learning 4.2 (1989): 227-243.

    Init Parameters
    ----------
    min_num : int, (default=2)
        decide the minimum number of each leaf, which will control the size of the tree

    raise_subtree: bool, (default=False)
        decide whether raise the subtree in the pruning process when possible, which will affect the
        size of the final decision tree.

    """

    def __init__(self, min_num=2, normalize=False, axis=0, raise_subtree=False):
        self.features = 0
        self.classes = 0
        self.root = dtm.TreeNode()
        self.stop_criterion = 0
        self.learned = False
        self.min_num = min_num
        self.normalize = normalize
        self.axis = axis
        self.raise_subtree = raise_subtree

    def fit(self, x, y):
        x = Normalizer.normalize(x, norm=self.normalize, axis=self.axis)
        instances = dtm.MLInstaces(x, y)
        self.features = instances.features
        self.classes = instances.classes

        # tests
        # if self.min_num <= self.classes:
        #     self.min_num = self.classes

        self.fit_tree(instances, self.root)
        countobj = dtm.Countobj()
        self.root.prune(self.raise_subtree, countobj)
        print("pruned: " + str(countobj.a))
        self.learned = True
        return self

    def fit_tree(self, data, treenode):
        if data.samples <= 2 * self.min_num or data.pure is True:
            treenode.leaf(data)
            return

        # select the best split point
        select_model = dtm.ModelSelection(use_mdl=True, min_num=self.min_num)
        pack = select_model.select(data)

        if pack is None:
            treenode.leaf(data)
            return

        attr, value = pack

        # split datasets for further splitting
        left_data, right_data, distribution = data.split(attr, value)

        treenode.distribution = distribution
        treenode.split_info = [attr, value]
        treenode.left = dtm.TreeNode()
        treenode.right = dtm.TreeNode()

        # continue learning
        self.fit_tree(left_data, treenode.left)
        self.fit_tree(right_data, treenode.right)

    def predict(self, x):
        if self.learned is False:
            raise Exception('this tree has not been fitted')

        x = check_feature_input(x)
        x = Normalizer.normalize(x, norm=self.normalize, axis=self.axis)
        samples, features = x.shape

        if features != self.features:
            raise Exception('inconsistent attribute number')

        results = []
        for index in range(samples):
            values = x[index]
            treenode = self.root
            # traverse the tree until reach the end
            while not treenode.is_leaf:
                attr_index, split_value = treenode.split_info
                if values[attr_index] <= split_value:
                    treenode = treenode.left
                else:
                    treenode = treenode.right
            results.append(treenode.get_prediected_labels())

        return results
