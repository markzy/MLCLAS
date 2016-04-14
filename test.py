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
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
test_data_trans = csr_matrix(pca.transform(test_data.todense()))

# job_server = pp.Server()


# start = time.time()
def train(a, b, c):
    return multilabel_algorithms.BPMLL(print_procedure=True,normalize=False).fit(a, b).predict(c)


# job1 = job_server.submit(train, args=(train_data_trans, train_target, test_data_trans), modules=('multilabel_algorithms',))
e = train(train_data_trans, train_target, test_data_trans)
em = models.EvaluationMetrics(data[3], e)
print(em.hamming_loss())
print(em.one_error())
# e = mlb.MultiLabelDecisionTree().fit(train_data_trans,train_target).predict(test_data_trans)
# print('It took {0:0.5f} seconds'.format(time.time() - start))

# file_name = 'results/BPMLL.pkl'
# with open(file_name, 'wb') as output:
#     pickle.dump(e, output, pickle.HIGHEST_PROTOCOL)
#
# exit()


