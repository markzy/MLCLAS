from sklearn import preprocessing
import numpy as np
from numpy import linalg as la


class Normalizer:
    @staticmethod
    def normalize(data, norm, axis=0):
        if norm is False:
            return data
        elif norm in ['l1', 'l2']:
            return preprocessing.normalize(data, norm=norm, axis=axis)
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
            norm_value = la.norm(data, ord=norm, axis=axis)
            if axis == 1:
                norm_value = np.asarray([norm_value]).T
            return data / norm_value
        else:
            raise Exception('Unknown type of normalization ' + str(norm))

# tests code
if __name__ == '__main__':
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(Normalizer.normalize(a,'l1',axis=1))
