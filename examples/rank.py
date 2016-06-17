import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from mlclas.neural import BPMLL
from mlclas.stats.metrics import RankMetrics

files = ['datasets/scene_train', 'datasets/scene_train']

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


result = BPMLL(print_procedure=True, neural=0.4, regularization=0, epoch=40, normalize='max').fit(train_data_trans, train_target).predict(test_data_trans, rank_metrics=True)

"""
    special metric for rank systems like BPMLL and RankingSVM, compatible with RankingSVM
    if you collect required information in predict() function of RankingSVM.
    See predict() function of BPMLL if you want to compute results in these metrics.
"""
metric = RankMetrics(test_target, result)
print('hamming loss:' + str(metric.hamming_loss()))
print('one error:' + str(metric.one_error()))
print('coverage:' + str(metric.coverage()))
print('ranking_loss:' + str(metric.ranking_loss()))
print('average_precision:' + str(metric.average_precision()))
print('precision:' + str(metric.precision()))
print('accuracy:' + str(metric.accuracy()))