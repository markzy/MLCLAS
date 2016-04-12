import pickle

firstN = 9

folder_path = 'data/Reuters/'
attribute_path = folder_path + 'first' + str(firstN) + '_data.pkl'
target_path = folder_path + 'first' + str(firstN) + '_target.pkl'
with open(attribute_path,'rb') as source:
    selected_tf_all = pickle.load(source)

with open(target_path, 'rb') as source:
    target = pickle.load(source)

print(selected_tf_all)
print('--------')
print(target)