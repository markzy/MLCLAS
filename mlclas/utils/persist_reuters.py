from peewee import *
import re
import string
from operator import itemgetter
import pickle

firstN = 9

db = MySQLDatabase('classification', host="localhost", user="root", passwd="root")

topic_dic = {0: 'earnings', 1: 'acquisitions', 2: 'commodity', 3: 'economics', 4: 'interest', 5: 'energy', 6: 'money-fx', 7: 'shipping',
             8: 'currency'}

with open("../datasets/topic_subcategory") as topic_source:
    lines = topic_source.read().split()

subcategory_dic = {}
# parent_pattern = re.compile('.*\((.*)\)')

index = 0
for line in lines:
    if line.isdigit():
        index = int(line)
    else:
        # mc = parent_pattern.match(line)
        # if mc:
        #     subcategory_dic[str.lower(mc.group(1))] = index
        # else:
        subcategory_dic[str.lower(line)] = index


class Reuters(Model):
    id = IntegerField()
    topics = CharField(200)
    body = CharField(15000)

    class Meta:
        database = db


db.connect()

with open("../datasets/stop_words.txt") as source:
    all_contents = source.read()
    regex = re.compile('\'')
    clear_up = regex.sub('\n\n', all_contents)
    stop_words = set(clear_up.split("\n\n"))

articles = Reuters.select()

digit_reg = re.compile('[0-9]+')
punctuation_reg = re.compile('[%s\n\t\r]' % re.escape(string.punctuation))

document_frequency = {}
tf_all = []
topics_all = []

# feature extraction
for article in articles:
    # all strings of digits are mapped to a single common token
    body = digit_reg.sub('rnums', article.body)
    # punctuation marks are removed
    punctuation_removed_content = punctuation_reg.sub(' ', body)
    # all words are converted to lower case
    final_content = str.lower(punctuation_removed_content)

    text_frequency = {}
    words = final_content.split(" ")
    for word in words:
        if word == '':
            continue
        if word in text_frequency:
            text_frequency[word] += 1
        elif word in document_frequency:
            text_frequency[word] = 1
            document_frequency[word] += 1
        else:
            text_frequency[word] = 1
            document_frequency[word] = 1

    tf_all.append(text_frequency)
    topics_all.append(article.topics.split(","))

# organize topics
organized_topics_all = []
for topics in topics_all:
    tmp = []
    for topic in topics:
        if topic in subcategory_dic and subcategory_dic[topic] not in tmp:
            tmp.append(subcategory_dic[topic])
    organized_topics_all.append(tmp)

with open("../datasets/reuters/category.pkl",'wb') as destination:
    pickle.dump(organized_topics_all, destination, pickle.HIGHEST_PROTOCOL)
    print('ok')

exit()

# feture selection
# remove stop words first
for sw in stop_words:
    if sw in document_frequency:
        del (document_frequency[sw])

# select 2% words with highest document frequency
word_num = len(document_frequency)
# ceil up
select_num = int(word_num * 0.02)
if select_num < word_num * 0.02:
    select_num += 1

sorted_output = sorted(document_frequency.items(), key=itemgetter(1), reverse=True)
selected_output = sorted_output[0:select_num]
# random.shuffle(selected_output)

# final selection
selected_df = {}
vocabulary = {}
for index in range(select_num):
    selected_df[selected_output[index][0]] = selected_output[index][1]
    vocabulary[selected_output[index][0]] = index

# compute bag of words
selected_tf_all = []
for single_tf in tf_all:
    selected_tf = [0 for i in range(select_num)]
    for word in single_tf:
        if word in vocabulary:
            selected_tf[vocabulary[word]] = single_tf[word]
    selected_tf_all.append(selected_tf)

folder_path = '../datasets/reuters/'
attribute_path = folder_path + 'first' + str(firstN) + '_data.pkl'
target_path = folder_path + 'first' + str(firstN) + '_target.pkl'
with open(attribute_path,'wb') as destination:
    pickle.dump(selected_tf_all, destination, pickle.HIGHEST_PROTOCOL)

with open(target_path, 'wb') as destination:
    pickle.dump(organized_topics_all, destination, pickle.HIGHEST_PROTOCOL)


