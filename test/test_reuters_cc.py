import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from mlclas.ensemble import ClassifierChains, MLKNN
from mlclas.stats import UniversalMetrics
from joblib import Parallel, delayed


def train_fuc(a, b, c, kk):
    # return ClassifierChains(LinearSVC()).fit(a, b).predict(c)
    return MLKNN(k=kk).fit(a, b).predict(c)


if __name__ == '__main__':
    data, target = datasets.load_svmlight_file('mlclas/data/reuters_all', multilabel=True)

    data = data.toarray()
    target_bi = np.array(MultiLabelBinarizer().fit_transform(target))
    target = np.array([list(k) for k in target])
    expected = []

    pro_num = 3

    kf = KFold(len(data), pro_num)
    for train, test in kf:
        expected.append(target[test])
    res = Parallel(n_jobs=-1)(delayed(train_fuc)(data[train], target_bi[train], data[test], 6) for train, test in kf)
    ems = [UniversalMetrics(expected[i], res[i]) for i in range(pro_num)]
    pr, ac = 0, 0
    for i in range(pro_num):
        pr += ems[i].precision() / pro_num
        ac += ems[i].accuracy() / pro_num

    print('precision:' + str(pr))
    print('accuracy:' + str(ac))
