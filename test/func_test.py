import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from mlclas.ensemble import ClassifierChains, MLKNN
from mlclas.stats import UniversalMetrics

files = ['mlclas/data/rtrain', 'mlclas/data/rtest']

data = datasets.load_svmlight_files(files, multilabel=True)
train_data = data[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
test_data = data[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))

# feature_size = train_data.shape[1]
# pca = PCA(n_components=(feature_size * 1) // 100)
# train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
# test_data_trans = csr_matrix(pca.transform(test_data.todense()))

e = ClassifierChains(LinearSVC()).fit(train_data, train_target).predict(test_data)
# op_array = [train_data_trans, train_target, test_data_trans, test_target]


# e = MLKNN(k=8).fit(train_data_trans, train_target).predict(test_data_trans)
# m = UniversalMetrics(data[3], e)
# print(m.precision())
# print(m.accuracy())
# datasets.dump_svmlight_file(train_data_trans, train_target, f='mlclas/data/rtrain', multilabel=True)
# datasets.dump_svmlight_file(test_data_trans, test_target, f='mlclas/data/rtest', multilabel=True)
