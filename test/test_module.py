import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from mlclas.neural.bpmll import BPMLL
from mlclas.neural import bpmll_models
from joblib import Parallel, delayed


def train_fuc(a, b, c):
    return BPMLL(print_procedure=False, epoch=40, normalize='max').fit(a, b).predict(c, True)


if __name__ == '__main__':
    files = ['mlclas/data/scene_train', 'mlclas/data/scene_test']

    data = datasets.load_svmlight_files(files, multilabel=True)
    train_data = data[0]
    train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
    test_data = data[2]
    test_target = np.array(MultiLabelBinarizer().fit_transform(data[3]))

    feature_size = train_data.shape[1]
    pca = PCA(n_components=(feature_size * 10) // 100)
    train_data_trans = csr_matrix(pca.fit_transform(train_data.todense())).toarray()
    test_data_trans = csr_matrix(pca.transform(test_data.todense())).toarray()

    pro_num = 10
    res = Parallel(n_jobs=-1)(delayed(train_fuc)(train_data_trans, train_target, test_data_trans) for i in range(pro_num))

    ems = [bpmll_models.BPMLLMetrics(data[3], res[i]) for i in range(pro_num)]
    hl, oe, cv, rl, ap, pr, ac = 0, 0, 0, 0, 0, 0, 0
    for i in range(pro_num):
        hl += ems[i].hamming_loss() / pro_num
        oe += ems[i].one_error() / pro_num
        cv += ems[i].coverage() / pro_num
        rl += ems[i].ranking_loss() / pro_num
        ap += ems[i].average_precision() / pro_num
        pr += ems[i].precision() / pro_num
        ac += ems[i].accuracy() / pro_num

    print('hamming loss:' + str(hl))
    print('one error:' + str(oe))
    print('coverage:' + str(cv))
    print('ranking_loss:' + str(rl))
    print('average_precision:' + str(ap))
    print('precision:' + str(pr))
    print('accuracy:' + str(ac))
