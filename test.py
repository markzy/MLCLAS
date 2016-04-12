from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np
import multilabel_algorithms as ml
import multilabel_algorithms_basic as mlb
import scipy
from scipy.sparse import csr_matrix
import pickle
import time
import models

files = ['data/scene_train', 'data/scene_test']

a = datasets.load_svmlight_files(files, multilabel=True)
train_data = a[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(a[1]))
test_data = a[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(a[3]))


feature_size = train_data.shape[1]
pca = PCA(n_components=(feature_size * 10) // 100)
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
test_data_trans = csr_matrix(pca.transform(test_data.todense()))

# start = time.time()
e = ml.BPMLL().fit(train_data_trans,train_target).predict(test_data_trans)
# e = mlb.MultiLabelDecisionTree().fit(train_data_trans,train_target).predict(test_data_trans)
# print('It took {0:0.5f} seconds'.format(time.time() - start))

file_name = 'results/BPMLL.pkl'
with open(file_name, 'wb') as output:
    pickle.dump(e, output, pickle.HIGHEST_PROTOCOL)

exit()
