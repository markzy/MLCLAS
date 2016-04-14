# from sklearn import datasets
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.svm import LinearSVC
# from sklearn.decomposition import PCA
# import numpy as np
# import multilabel_algorithms
# import ml_old
# from scipy.sparse import csr_matrix
# import pickle
# import models
# import pp
#
# files = ['data/scene_train', 'data/scene_test']
#
# data = datasets.load_svmlight_files(files, multilabel=True)
# train_data = data[0]
# train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
# test_data = data[2]
# test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))
#
# feature_size = train_data.shape[1]
# pca = PCA(n_components=(feature_size * 10) // 100)
# train_data_trans = np.array(pca.fit_transform(train_data.todense()))
# test_data_trans = np.array(pca.transform(test_data.todense()))
#
# ml = multilabel_algorithms.BPMLL(normalize=False).init(train_data_trans, train_target)
# oldml = ml_old.BPMLL().init(train_data_trans, train_target)
#
# ml.wsj_matrix = np.array(oldml.wsj)
# ml.vhs_matrix = np.array(oldml.vhs)
#
# g1 = ml.fit_once(0)
# g2 = oldml.fit_once(0)
#
# print('hh')