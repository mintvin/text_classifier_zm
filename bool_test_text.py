#encoding=utf-8

import jieba
import  re
import sys
import json
import pickle as pk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def text_process(file_path,stopword_list):
    data_list = []
    class_list = []
    word_list = []
    for line in open(file_path,'r'):
        line = line.decode('utf-8').strip()
        # line = line.strip()
        classes = line.split('\t')[0]
        content = line.split('\t')[1]
        rule = re.compile(u'[^\u4E00-\u9FA5]')
        content = rule.sub(r'',content)
        contents = list(jieba.cut(content,cut_all=False))
        contents = del_stopword(contents,stopword_list)
        # word_string = " ".join(contents).encode('utf-8')
        # print word_string
        # print " "
        # print classes
        data_list.append(contents)
        # data_list.append(contents)
        class_list.append(classes.encode('utf-8'))
    # print 'c'
    return data_list,class_list

def del_stopword(data_list,stopword_list):
    contents = []
    for word in data_list:
        if word not in stopword_list:
            contents.append(word)
    return contents

def get_stopword(stopword_path):
    # words_set = set()
    stop_words = []
    with open(stopword_path, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in stop_words:
                stop_words.append(word)
    return stop_words

def get_featureword(word_path):
    feature_word = []
    for line in open(word_path,'r'):
        line = line.decode('utf-8').strip()
        feature_word.append(line)
    return feature_word

def text_feature(test_data_list,feature_word):

    def text_feature(data,feature_word):
        data_words = set(data)
        feature = [1 if word in data_words else 0 for word in feature_word]
        # print "a"
        # print feature
        return feature

    test_feature = [text_feature(data,feature_word) for data in test_data_list]
    # print train_feature,test_feature
    return test_feature

def chi_feature(test_feature,class_list,faeture_word):
    chi_model = joblib.load('./model/chi_model.pkl')
    chi_feature = chi_model.fit_transform(test_feature,class_list)
    feature_name = [feature_word[i] for i in chi_model.get_support(indices=True)]
    print json.dumps(feature_name,encoding='utf-8',ensure_ascii=False)
    return chi_feature


def text_classifier(test_feature,test_class_list):
    target = ['体育','娱乐','家居','教育','时尚','时政','游戏','科技','财经','房产']

    print json.dumps(test_class_list,encoding='utf-8',ensure_ascii=False)
    # for classes in test_class_list:
        # cls.append(classes.en)
        # print classes.decode('utf-8')
    classifier = joblib.load('./model/bool_classifier.pkl')
    print "test"
    test_predict = classifier.predict(test_feature)
    # print test_predict
    for word in test_predict:
        print word

    test_accuracy = classifier.score(test_feature,test_class_list)
    print "准确率", test_accuracy
    print"每个类别的精确率和召回率："
    print  classification_report(test_class_list, test_predict)
    return test_accuracy


if __name__ == '__main__':
    reload(sys)

    sys.setdefaultencoding('utf8')
    # file = './cnews/cnews.val.txt'
    file = './cnews/test_text.txt'
    word_path = './feature_word.txt'
    stopword_path = './cnews/stopwords.txt'

    stopword_list = get_stopword(stopword_path)


    test_data_list,test_class_list = text_process(file,stopword_list)
    feature_word = get_featureword(word_path)
    test_feature = text_feature(test_data_list,feature_word)
    chi_feature = chi_feature(test_feature,test_class_list,feature_word)

    test_accuracy = text_classifier(chi_feature,test_class_list)

    print "finished"









