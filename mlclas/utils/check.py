import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import MultiLabelBinarizer


def check_feature_input(array):
    """ check whether the input array is valid and return required form of input """
    if issparse(array):
        x_array = array.toarray()
    else:
        x_array = np.array(array)

    if x_array.ndim != 2:
        raise Exception('input feature matrix is not a valid 2D array')

    if not np.issubdtype(x_array.dtype, int) and not np.issubdtype(x_array.dtype, float):
        raise Exception('datasets in the input matrix is neither int or float')

    return x_array


def check_target_input(array):
    """
    check whether array is a valid target matrix
    :param array: array-like object
    :return: valid target matrix
    """
    if issparse(array):
        y_array = array.toarray()
    else:
        y_array = np.array(array)

    if y_array.ndim == 1:
        y_array = MultiLabelBinarizer().fit_transform(y_array)
    elif y_array.ndim == 2:
        flat_y = y_array.flatten()
        # check if it is already binarized
        for value in flat_y:
            if value != 0 and value != 1:
                y_array = MultiLabelBinarizer().fit_transform(y_array)
                break
    return y_array


# some tests code
if __name__ == '__main__':
    print(check_feature_input([[1, 2], [1, 3]]))
    print(check_target_input([{'scipy', 'numpy'}, {'sklearn'}]))

    # print(check_feature_input([[1, 2], [1]]))
    print(check_feature_input([['hah', 'hah'], ['hah', 'hah']]))
