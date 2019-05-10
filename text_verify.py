#encoding =utf-8

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import re
import random
import json
from sklearn.naive_bayes import MultinomialNB


def text_process(file_path):
    data_list = []
    with open(file_path,'r') as fp:
        content = fp.read().decode('utf-8').strip()
        rule = re.compile(u'[^\u4E00-\u9FA5]')
        content = rule.sub(r'',content)
        word_list = list(jieba.cut(content,cut_all=False))
        word_string = " ".join(word_list).encode('utf-8')
        data_list.append(word_string)

    return data_list


def get_stopword(stopword_path):
    # words_set = set()
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


def text_classifier(val_feature):
    classifier = joblib.load('./model/classifier.pkl')
    predict = classifier.predict(val_feature)
    # test_accuracy = classifier.score(val_feature,val_class_list)
    for word in predict:
        print word
    return predict

if __name__ == '__main__':
    val_file = './cnews/339789.txt'
    stopword_path = './cnews/stopwords.txt'

    val_data_list= text_process(val_file)
    stopword_list = get_stopword((stopword_path))


    val_feature = text_feature(val_data_list,stopword_list)
    val_accuracy = text_classifier(val_feature)

    print "finished"
