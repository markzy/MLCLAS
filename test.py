from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np
import multi_label as ml
from scipy.sparse import csr_matrix
import pickle

# files = ['/Users/Mark/Downloads/rcv1subset_topics_train_1.svm','/Users/Mark/Downloads/rcv1subset_topics_test_1.svm']
files = ['/Users/Mark/Downloads/scene_train', '/Users/Mark/Downloads/scene_test']

a = datasets.load_svmlight_files(files, multilabel=True)
train_data = a[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(a[1]))
test_data = a[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(a[3]))

# feature_size = train_data.shape[0]
# pca = PCA(n_components=(feature_size * 5) // 100)
# train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
# test_data_trans = csr_matrix(pca.transform(test_data.todense()))


e = ml.BinaryRelevance(LinearSVC(random_state=0)).fit(train_data, train_target).predict(test_data)
print(e)

with open('BinaryRelevance.pkl', 'wb') as output:
    pickle.dump(e, output, pickle.HIGHEST_PROTOCOL)

# def accuracy(y_ture, y_pred):

