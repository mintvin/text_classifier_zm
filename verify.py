#encoding =utf-8

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import re
import random
import json
from sklearn.naive_bayes import MultinomialNB


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

def text_feature(val_data_list,stopword_list):
    tfidf_vec = joblib.load('./model/tfidf_vec.pkl')
    chi_model = joblib.load('./model/chi_model.pkl')
    feature = tfidf_vec.get_feature_names()
    vocabulary = [feature[i] for i in chi_model.get_support(indices=True)]
    # vocabulary = tfidf_vec.vocabulary_
    vec = TfidfVectorizer(stop_words=stopword_list,sublinear_tf=True, max_df=0.5,vocabulary=vocabulary,max_features=1000)
    # voc = vec.vocabulary_
    # print json.dumps(voc,encoding='utf-8',ensure_ascii=False)
    val_feature = vec.fit_transform(val_data_list)
    # val = chi_model.fit_transform(val_feature,val_class_list)
    # print (val_feature.toarray()==val.toarray()).all()
    # print (test.toarray()==test_feature.toarray()).all()

    return val_feature


def text_classifier(val_feature,val_class_list):
    classifier = joblib.load('./model/classifier.pkl')
    predict = classifier.predict(val_feature)
    # predict = predict.tolist()
    # print json.dumps(predict,encoding='utf-8',ensure_ascii=False)
    # print json.dumps(val_class_list,encoding='utf-8',ensure_ascii=False)
    test_accuracy = classifier.score(val_feature,val_class_list)
    print "val",test_accuracy
    print "每个类别的精确率和召回率："
    print classification_report(val_class_list,predict)
    return test_accuracy

if __name__ == '__main__':
    val_file = './cnews/cnews.val.txt'
    stopword_path = './cnews/stopwords.txt'

    val_data_list,val_class_list = text_process(val_file)

    stopword_list = get_stopword((stopword_path))

    val_feature = text_feature(val_data_list,stopword_list)
    val_accuracy = text_classifier(val_feature,val_class_list)


    print "finished"
