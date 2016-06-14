import pickle
import time
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from mlclas.neural.bpmll import BPMLL
from mlclas.neural import bpmll_models
from joblib import Parallel, delayed


def train_fuc(a, b, c):

    return BPMLL(neural=0.2, epoch=40, normalize='max', axis=1, print_procedure=True).fit(a, b).predict(c, True)


if __name__ == '__main__':
    file_name = '../../data/reuters/first9_data.pkl'
    with open(file_name, 'rb') as input_:
        data = pickle.load(input_)

    file_name = '../../data/reuters/first9_target.pkl'
    with open(file_name, 'rb') as input_:
        target = pickle.load(input_)

    target = np.array(target)
    data = np.array(data, dtype='float64')
    target_bi = np.array(MultiLabelBinarizer().fit_transform(target))
    expected = []

    pro_num = 3

    kf = KFold(len(data), pro_num)
    res = Parallel(n_jobs=-1)(delayed(train_fuc)(data[train], target_bi[train], data[test]) for train, test in kf)

    for train, test in kf:
        expected.append(target[test])

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

    ems = [bpmll_models.BPMLLMetrics(expected[i], res[i]) for i in range(pro_num)]
    hl, oe, cv, rl, ap, pr, ac = 0, 0, 0, 0, 0, 0, 0
    for i in range(pro_num):
        hl += ems[i].hamming_loss() / pro_num
        oe += ems[i].one_error() / pro_num
        cv += ems[i].coverage() / pro_num
        rl += ems[i].ranking_loss() / pro_num
        ap += ems[i].average_precision() / pro_num
        pr += ems[i].precision() / pro_num
        ac += ems[i].accuracy() / pro_num

    # with open('../results/Reuters/first9_result', 'w') as output_:
    #     output_.write('hamming loss:' + str(hl) + '\n')
    #     output_.write('one error:' + str(oe) + '\n')
    #     output_.write('coverage:' + str(cv) + '\n')
    #     output_.write('ranking_loss:' + str(rl) + '\n')
    #     output_.write('average_precision:' + str(ap) + '\n')
    #     output_.write('It took {0:0.5f} seconds'.format(learn_time))

    print('hamming loss:' + str(hl))
    print('one error:' + str(oe))
    print('coverage:' + str(cv))
    print('ranking_loss:' + str(rl))
    print('average_precision:' + str(ap))
    print('precision:' + str(pr))
    print('accuracy:' + str(ac))
