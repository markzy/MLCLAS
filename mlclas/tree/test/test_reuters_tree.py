import time
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from mlclas.tree import MLDecisionTree
from mlclas.stats import UniversalMetrics
from joblib import Parallel, delayed


def train_fuc(a, b, c):
    return MLDecisionTree(normalize='max', axis=1).fit(a, b).predict(c)


if __name__ == '__main__':
    data, target = datasets.load_svmlight_file('../../data/reuters_all',multilabel=True)

    data = data.toarray()
    target_bi = np.array(MultiLabelBinarizer().fit_transform(target))
    target = np.array([list(k) for k in target])
    expected = []

    pro_num = 3

    kf = KFold(len(data), pro_num)
    for train, test in kf:
        expected.append(target[test])

    res = Parallel(n_jobs=-1)(delayed(train_fuc)(data[train], target_bi[train], data[test]) for train, test in kf)

    # file_name = '../../results/bpmll/result.pkl'
    # with open(file_name, 'wb') as output_:
    #     pickle.dump(result, output_, pickle.HIGHEST_PROTOCOL)
    #
    # file_name = '../../results/bpmll/expected.pkl'
    # with open(file_name, 'wb') as output_:
    #     pickle.dump(expected, output_, pickle.HIGHEST_PROTOCOL)

    # print('result has been serialized to local file')

    # file_name = 'results/BPMLL/result.pkl'
    # with open(file_name, 'rb') as input_:
    #     result = pickle.load(input_)
    #
    # file_name = 'results/BPMLL/expected.pkl'
    # with open(file_name, 'rb') as input_:
    #     expected = pickle.load(input_)

    ems = [UniversalMetrics(expected[i], res[i]) for i in range(pro_num)]
    pr, ac = 0, 0
    for i in range(pro_num):
        pr += ems[i].precision() / pro_num
        ac += ems[i].accuracy() / pro_num

    # with open('../results/Reuters/first9_result', 'w') as output_:
    #     output_.write('hamming loss:' + str(hl) + '\n')
    #     output_.write('one error:' + str(oe) + '\n')
    #     output_.write('coverage:' + str(cv) + '\n')
    #     output_.write('ranking_loss:' + str(rl) + '\n')
    #     output_.write('average_precision:' + str(ap) + '\n')
    #     output_.write('It took {0:0.5f} seconds'.format(learn_time))

    print('precision:' + str(pr))
    print('accuracy:' + str(ac))
