import pickle
import models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import KFold
import multilabel_algorithms as ml
import numpy as np
import time

file_name = 'data/Reuters/first9_data.pkl'
with open(file_name, 'rb') as input_:
    data = pickle.load(input_)

file_name = 'data/Reuters/first9_target.pkl'
with open(file_name, 'rb') as input_:
    target = pickle.load(input_)

data = np.array(data)
target = np.array(MultiLabelBinarizer().fit_transform(target))

kf = KFold(len(data), 3)
result = []
expected = []

start = time.time()
for train, test in kf:
    result.append(ml.BPMLL(regulization=0.1).fit(data[train], target[train]).predict(data[test]))
    expected.append(target[test])
learn_time = time.time() - start


file_name = 'results/BPMLL.pkl'
with open(file_name, 'wb') as output_:
    pickle.dump(result, output_, pickle.HIGHEST_PROTOCOL)

ems = [models.EvaluationMetrics(expected[0],result[0]), models.EvaluationMetrics(expected[1],result[1]), models.EvaluationMetrics(expected[2],result[2])]
hl ,oe, cv, rl, ap = 0
for i in range(3):
    hl += ems[i].hamming_loss()/3
    oe += ems[i].one_error()/3
    cv += ems[i].coverage()/3
    rl += ems[i].ranking_loss()/3
    ap += ems[i].average_precision()/3

with open('results/Reuters/first9_result','w') as output_:
    output_.write('hamming loss:' + str(hl) + '\n')
    output_.write('one error:' + str(oe) + '\n')
    output_.write('coverage:' + str(cv) + '\n')
    output_.write('ranking_loss:' + str(rl) + '\n')
    output_.write('average_precision:' + str(ap) + '\n')
    output_.write('It took {0:0.5f} seconds'.format(learn_time))