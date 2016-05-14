from sklearn import preprocessing

X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2', axis=0)

print(X_normalized)
