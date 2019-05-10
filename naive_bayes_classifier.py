#encoding =utf-8

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import sys
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import re
import random
import json
from sklearn.naive_bayes import MultinomialNB

reload(sys)
sys.setdefaultencoding('utf-8')


def text_process(file_path):
    data_list = []
    class_list = []
    for line in open(file_path,'r'):
        line = line.decode('utf-8').strip()
        classes = line.split('\t')[0]
        content = line.split('\t')[1]
        rule = re.compile(u'[^\u4E00-\u9FA5]')
        content = rule.sub(r'',content)
        word_list = list(jieba.cut(content,cut_all=False))
        word_string = " ".join(word_list).encode('utf-8')
        data_list.append(word_string)
        class_list.append(classes.encode('utf-8'))

    return data_list,class_list


def get_stopword(stopword_path):
    stop_words = []
    with open(stopword_path, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in stop_words:
                stop_words.append(word)
    return stop_words

def text_feature(train_data_list,test_data_list,stopword_list):
    tfidf = TfidfVectorizer(stop_words=stopword_list,sublinear_tf=True, max_df=0.5)
    chi1 = SelectKBest(chi2,k=1000)
    train_feature = tfidf.fit_transform(train_data_list)
    print train_feature.shape
    feature = tfidf.get_feature_names()
    train_feature = chi1.fit_transform(train_feature,train_class_list)
    # vocabulary = tfidf.vocabulary_
    feature_name = [feature[i] for i in chi1.get_support(indices=True)]
    vocabulary = feature_name
    print len(vocabulary)

    joblib.dump(tfidf,'./model/tfidf_vec.pkl')
    joblib.dump(chi1,'./model/chi_model.pkl')
    vec = TfidfVectorizer(stop_words=stopword_list,sublinear_tf=True, max_df=0.5,vocabulary=vocabulary,max_features=1000)
    chi = SelectKBest(chi2,k=1000)
    # voc = vec.vocabulary_
    # print json.dumps(voc,encoding='utf-8',ensure_ascii=False)
    test_feature = vec.fit_transform(test_data_list)
    print test_feature.shape
    # test_feature = chi.fit_transform(test_feature,test_class_list)
    return train_feature,test_feature

def text_classifier(train_feature,train_class_list,test_feature,test_class_list):
    clf = MultinomialNB()
    classifier = clf.fit(train_feature,train_class_list)
    train_score = cross_val_score(clf,train_feature,train_class_list,cv=10,scoring='accuracy')
    joblib.dump(clf,'./model/classifier.pkl')

    train_score = train_score .mean()
    print train_score
    test_predict = clf.predict(test_feature)
    test_accuracy = classifier.score(test_feature,test_class_list)
    print test_accuracy
    print"每个类别的精确率和召回率："
    print  classification_report(test_class_list, test_predict)
    return test_accuracy,train_score



if __name__ == '__main__':
    train_file = './cnews/cnews.train.txt'
    test_file = './cnews/cnews.test.txt'
    stopword_path = './cnews/stopwords.txt'

    train_data_list,train_class_list = text_process(train_file)
    test_data_list,test_class_list = text_process(test_file)

    stopword_list = get_stopword((stopword_path))

    train_feature,test_feature= text_feature(train_data_list,test_data_list,stopword_list)

    test_accuracy,train_score = text_classifier(train_feature,train_class_list,test_feature,test_class_list)

    print "finished"
