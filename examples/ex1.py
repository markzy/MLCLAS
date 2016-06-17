import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from mlclas.ensemble import BinaryRelevance, ClassifierChains, CalibratedLabelRanking, RandomKLabelsets, MLKNN
from mlclas.tree import MLDecisionTree
from mlclas.neural import BPMLL
from mlclas.svm import RankingSVM
from mlclas.stats import UniversalMetrics

files = ['datasets/scene_train', 'datasets/scene_test']

# load files
data = datasets.load_svmlight_files(files, multilabel=True)
train_data = data[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
test_data = data[2]
test_target = data[3]

# feature extraction using PCA
feature_size = train_data.shape[1]
pca = PCA(n_components=(feature_size * 10) // 100)
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
test_data_trans = csr_matrix(pca.transform(test_data.todense()))

"""
    train and predict using any of following scripts:

    1.  result = BinaryRelevance(LinearSVC()).fit(train_data, train_target).predict(test_data)

    2.  result = ClassifierChains(LinearSVC()).fit(train_data, train_target).predict(test_data)

    3.  result = CalibratedLabelRanking(LinearSVC()).fit(train_data, train_target).predict(test_data)

    4.  result = RandomKLabelsets(LinearSVC()).fit(train_data, train_target).predict(test_data)

    5.  result = MLKNN(any integer, for example 6).fit(train_data, train_target).predict(test_data)

    6.  result = MLDecisionTree(min_num=10).fit(train_data_trans, train_target).predict(test_data_trans)

    7.  result = BPMLL(print_procedure=True, neural=0.4, regularization=0, epoch=40, normalize='max').fit(train_data_trans, train_target)
                .predict(test_data_trans, use_metrics=False)

    8.  result = RankingSVM(normalize='l2', print_procedure=True).fit(train_data_trans, train_target, 8).predict(test_data_trans)

"""

result = MLDecisionTree(min_num=10).fit(train_data_trans, train_target).predict(test_data_trans)

# metrics
m = UniversalMetrics(test_target, result)
print('precision: ' + str(m.precision()))
print('accuracy: ' + str(m.accuracy()))
