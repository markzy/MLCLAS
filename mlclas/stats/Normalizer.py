from sklearn import preprocessing
import numpy as np
from numpy import linalg as la


class Normalizer:
    @staticmethod
    def normalize(data, norm):
        if norm is False:
            return data
        elif norm in ['l1', 'l2']:
            return preprocessing.normalize(data, norm=norm, axis=0)
        elif norm == 'fs':
            samples, features = data.shape
            min_value, max_value = (0, 1)
            for i in range(features):
                array_min = np.min(data[:, i])
                array_max = np.max(data[:, i])
                data[:, i] = ((data[:, i] - array_min) / (array_max - array_min) * (max_value - min_value)) + min_value
            return data
        elif norm in ['max', 'min']:
            dic = {'max': np.inf, 'min': -np.inf}
            norm = dic[norm]
            norm_value = la.norm(data, ord=norm, axis=0)
            return data / norm_value
        else:
            raise Exception('Unknown type of normalization ' + str(norm))

# test code
if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    print(Normalizer.normalize(a,'min'))
