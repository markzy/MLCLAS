from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np
import multilabel_algorithms
from scipy.sparse import csr_matrix
import pickle
import models
import pp

files = ['data/scene_train', 'data/scene_test']

data = datasets.load_svmlight_files(files, multilabel=True)
train_data = data[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
test_data = data[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))

feature_size = train_data.shape[1]
pca = PCA(n_components=(feature_size * 10) // 100)
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense())).toarray()
test_data_trans = csr_matrix(pca.transform(test_data.todense())).toarray()

e = multilabel_algorithms.BPMLL(print_procedure=True,epoch=40).fit(train_data_trans, train_target)
res = e.predict(test_data_trans)




em = models.EvaluationMetrics(data[3], res)
print(em.predictedLabels)
print('----------')
print(data[3])
print('hamming loss:' + str(em.hamming_loss()))
print('one error:' + str(em.one_error()))
print('coverage:' + str(em.coverage()))
print('ranking_loss:' + str(em.ranking_loss()))
print('average_precision:' + str(em.average_precision()))
