import time
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from mlclas.neural.bpmll import BPMLL
from mlclas.neural import bpmll_models
from mlclas.stats import UniversalMetrics
from mlclas.svm.ranking_svm import RankingSVM
from mlclas.tree import MLDecisionTree
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from sklearn.cross_validation import KFold


def get_yeast():
    files = ['mlclas/data/yeast_train', 'mlclas/data/yeast_test']
    data = datasets.load_svmlight_files(files, multilabel=True)
    return data


def get_reuters():
    file = 'mlclas/data/reuters_all'
    data = datasets.load_svmlight_file(file, multilabel=True)
    return data


def pca_pro(train, test, percent):
    feature_size = train.shape[1]
    pca = PCA(n_components=(feature_size * percent) // 100)
    train_trans = csr_matrix(pca.fit_transform(train.todense())).toarray()
    test_trans = csr_matrix(pca.transform(test.todense())).toarray()
    return train_trans, test_trans


def fit_tree(a, b, c, min_num=2, normalize=False, axis=0):
    return MLDecisionTree(normalize=normalize, min_num=min_num, axis=axis).fit(a, b).predict(c)


def fit_neural(a, b, c, neural=0.2, normalize=False, axis=0):
    return BPMLL(neural=neural, epoch=100, normalize=normalize, axis=axis).fit(a, b).predict(c)


def fit_svm(a, b, c, c_factor, normalize=False, axis=0):
    return RankingSVM(normalize=normalize, axis=axis, print_procedure=True).fit(a, b, c_factor).predict(c)


def yeast_bench(func):
    data = get_yeast()
    train_data = data[0]
    train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
    test_data = data[2]
    # test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))
    train_data_trans, test_data_trans = pca_pro(train_data, test_data, 50)

    pro_num = 0
    if func == 'tree':
        pro_num = 7
        res = Parallel(n_jobs=-1)(delayed(fit_tree)(train_data_trans, train_target, test_data_trans, k, 'max', 0) for k in [2, 5, 7, 10, 15, 20, 25])
    elif func == 'neural':
        pro_num = 4
        res = Parallel(n_jobs=-1)(delayed(fit_neural)(train_data_trans, train_target, test_data_trans, k, 'max', 0) for k in [0.2, 0.4, 0.6, 0.8])
    else:
        pro_num = 6
        res = Parallel(n_jobs=-1)(delayed(fit_svm)(train_data_trans, train_target, test_data_trans, k, 'l2', 0) for k in [7, 10, 14, 20, 24, 30])

    ems = [UniversalMetrics(data[3], res[i]) for i in range(pro_num)]
    pr = [ems[i].precision() for i in range(pro_num)]
    ac = [ems[i].accuracy() for i in range(pro_num)]
    print(str(pr))
    print(str(ac))


def reuters_bench(func):
    data, target = get_reuters()
    data = data.toarray()
    target_bi = np.array(MultiLabelBinarizer().fit_transform(target))
    target = np.array([list(k) for k in target])
    expected = []

    pro_num = 3

    kf = KFold(len(data), pro_num)
    for train, test in kf:
        expected.append(target[test])

    pr_list = []
    ac_list = []

    if func == 'tree':
        for k in [2, 5, 7, 10, 15, 20, 25]:
            res = Parallel(n_jobs=-1)(delayed(fit_tree)(data[train], target_bi[train], data[test], k, 'max', 1) for train, test in kf)
            ems = [UniversalMetrics(expected[i], res[i]) for i in range(pro_num)]
            pr, ac = 0, 0
            for i in range(pro_num):
                pr += ems[i].precision() / pro_num
                ac += ems[i].accuracy() / pro_num
            pr_list.append(pr)
            ac_list.append(ac)
            print('k is now ' + str(k))
            print(pr)
            print(ac)

    if func == 'neural':
        for k in [0.2, 0.4, 0.6, 0.8]:
            res = Parallel(n_jobs=-1)(delayed(fit_svm)(data[train], target_bi[train], data[test], k, 'max', 1) for train, test in kf)
            ems = [UniversalMetrics(expected[i], res[i]) for i in range(pro_num)]
            pr, ac = 0, 0
            for i in range(pro_num):
                pr += ems[i].precision() / pro_num
                ac += ems[i].accuracy() / pro_num
            pr_list.append(pr)
            ac_list.append(ac)
            print('k is now ' + str(k))
            print(pr)
            print(ac)

    if func == 'svm':
        for k in [6]:
            res = Parallel(n_jobs=-1)(delayed(fit_svm)(data[train], target_bi[train], data[test], k, 'max', 1) for train, test in kf)
            ems = [UniversalMetrics(expected[i], res[i]) for i in range(pro_num)]
            pr, ac = 0, 0
            for i in range(pro_num):
                pr += ems[i].precision() / pro_num
                ac += ems[i].accuracy() / pro_num
            pr_list.append(pr)
            ac_list.append(ac)
            print('k is now ' + str(k))
            print(pr)
            print(ac)

    # for train, test in kf:
    #     fit_svm(data[train], target_bi[train], data[test], 6, 'max', 1)
    #     break

    print(str(pr_list))
    print(str(ac_list))


if __name__ == '__main__':
    # yeast_bench('tree')
    reuters_bench('svm')
