import pickle
import models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import KFold
import multilabel_algorithms as ml
import numpy as np

file_name = 'data/Reuters/first9_data.pkl'
with open(file_name, 'rb') as input_:
    data = pickle.load(input_)

file_name = 'data/Reuters/first9_target.pkl'
with open(file_name, 'rb') as input_:
    target = pickle.load(input_)

data = np.array(data)
target = np.array(MultiLabelBinarizer().fit_transform(target))

kf = KFold(len(data), 3)
result = None
for train, test in kf:
    result = ml.BPMLL(regulization=0.1).fit(data[train], target[train]).predict(data[test])
    break

file_name = 'results/BPMLL.pkl'
with open(file_name, 'wb') as output_:
    pickle.dump(result, output_, pickle.HIGHEST_PROTOCOL)
