import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
import pickle as pk
from mlclas.utils.metrics import UniversalMetrics
from mlclas.svm.ranking_svm import RankingSVM

files = ['./data/scene_train', './data/scene_test']

data = datasets.load_svmlight_files(files, multilabel=True)
train_data = data[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
test_data = data[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))

feature_size = train_data.shape[1]
pca = PCA(n_components=(feature_size * 5) // 100)
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense())).toarray()
test_data_trans = csr_matrix(pca.transform(test_data.todense())).toarray()

train_data_trans = preprocessing.normalize(train_data_trans, norm='l2', axis=0)
test_data_trans = preprocessing.normalize(test_data_trans, norm='l2', axis=0)

a = RankingSVM(True).fit(train_data_trans, train_target, 20)

# with open('./results/SVM/RSVM.pkl', 'rb') as _input:
#     rsvm = pk.load(_input)

e = a.predict(test_data_trans)
m = UniversalMetrics(6, data[3], e)
print(m.precision())
print(m.accuracy())
