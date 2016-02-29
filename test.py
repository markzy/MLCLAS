from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np
import multi_label_algorithms as ml
from scipy.sparse import csr_matrix
import pickle
import time

files = ['data/scene_train', 'data/scene_test']

a = datasets.load_svmlight_files(files, multilabel=True)
train_data = a[0]
train_target = np.array(MultiLabelBinarizer().fit_transform(a[1]))
test_data = a[2]
test_target = np.array(MultiLabelBinarizer().fit_transform(a[3]))


feature_size = train_data.shape[0]
pca = PCA(n_components=(feature_size * 5) // 100)
train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
test_data_trans = csr_matrix(pca.transform(test_data.todense()))

start = time.time()
e = ml.MultiLabelDecisionTree().fit(train_data_trans, train_target).predict(test_data_trans)
print('It took {0:0.1f} seconds'.format(time.time() - start))
print(e)

file_name = 'results/MultiLabelDecisionTree.pkl'
with open(file_name, 'wb') as output:
    pickle.dump(e, output, pickle.HIGHEST_PROTOCOL)

with open(file_name, 'rb') as input_:
    result = pickle.load(input_)


predicted_size = len(result)
accuracy = 0
for i in range(predicted_size):
    true_labels = a[3][i]
    predicted_labels = e[i]
    intersection = 0
    sum = len(predicted_labels)
    for label in true_labels:
        if label in predicted_labels:
            intersection += 1
        else:
            sum += 1
    accuracy += intersection/sum
accuracy /= predicted_size

print("accuracy:  ", accuracy)