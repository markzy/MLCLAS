import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

with open("mlclas/data/reuters/category.pkl", 'rb') as source:
    topics = pickle.load(source)

test_target = np.array(MultiLabelBinarizer().fit_transform(topics))
sum_all = np.sum(test_target, axis=0)

tmp_sum = []
portion = []
for k in range(1, 9):
    sum_samples = np.sum(test_target[:, :k+1], axis=1)
    label_sum = np.sum(sum_samples)
    sample_count = 0
    for num in sum_samples:
        if num > 0:
            sample_count += 1
    tmp_sum.append(sample_count)
    portion.append(label_sum / sample_count)

print(sum_all)
print(tmp_sum)
print(portion)


