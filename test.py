#encoding=utf-8

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.datasets import load_digits
import json

'''
train = [['中国 体育 篮球'],['小区 草地 位置'],['电影 演技 获奖']]
target = ['体育','房产','娱乐']
model_chi = SelectKBest(chi2,k=1)
    # print data_list
train = np.array(train)
text_feature = model_chi.fit_transform(train,target)
print text_feature
print text_feature.shape
    # return text_feature
'''
count_list =[]
feature_list = []
for line in open('./sorted_count_feature.txt','r'):
    count_list.append(line.strip('\n'))


print json.dumps(count_list,encoding='utf-8',ensure_ascii=False)

for line in open('./count_feature.txt','r'):
    feature_list.append(line.strip('\n'))
print json.dumps(feature_list,encoding='utf-8',ensure_ascii=False)

word_list = []
word_list2 = []
for word in feature_list:
    if word  not in count_list:
        # print word
        word_list.append(word)
print len(word_list)
print json.dumps(word_list,encoding='utf-8',ensure_ascii=False)


for word in count_list:
    if word  not in feature_list:
        # print word
        word_list2.append(word)
print len(word_list2)
print json.dumps(word_list2,encoding='utf-8',ensure_ascii=False)




