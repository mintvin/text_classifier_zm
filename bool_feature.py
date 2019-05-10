#encoding=utf-8

import jieba
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import re
from sklearn.naive_bayes import MultinomialNB
import sys
import json
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def text_process(file_path,stopword_list):
    data_list = []
    class_list = []
    word_list = []
    for line in open(file_path,'r'):
        line = line.decode('utf-8').strip()
        classes = line.split('\t')[0]
        content = line.split('\t')[1]
        rule = re.compile(u'[^\u4E00-\u9FA5]')
        content = rule.sub(r'',content)
        contents = list(jieba.cut(content,cut_all=False))
        contents = del_stopword(contents,stopword_list)
        # word_string = " ".join(word_list).encode('utf-8')
        # print word_string
        # print " "
        # print classes
        # data_list.append(word_string)
        data_list.append(contents)
        class_list.append(classes.encode('utf-8'))

    return data_list,class_list

def del_stopword(data_list,stopword_list):
    contents = []
    for word in data_list:
        if word not in stopword_list:
            contents.append(word)
    return contents

def word_frequence_dict(train_data_list):
    word_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word_dict.has_key(word):
                word_dict[word]  +=1
            else:
                word_dict[word] =1
    word_tuple_list = sorted(word_dict.items(),key= lambda f:f[1],reverse=True)
    word_list = list(zip(*word_tuple_list)[0])
    # with open ('./word_list.txt','w') as fp:
    #     for word in word_list:
    #         fp.write(word.encode('utf-8')+'\n')
            # fp.writelines(word.encode('utf-8'))
    return word_list



def get_feature_word(word_list,num,stopword_list):
    feature_word = []
    number =1
    for n in range(num,len(word_list),1):
        if number>1000:
            break
        if  1<len(word_list[n]) < 5:
            feature_word.append(word_list[n])
            number+=1
    print len(feature_word)
    with open('./feature_word.txt','w') as fp:
        for word in feature_word:
            fp.write(word.encode('utf-8')+'\n')
        # fp.write('---------------------------------')

    return feature_word

def get_stopword(stopword_path):
    # words_set = set()
    stop_words = []
    with open(stopword_path, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in stop_words:
                stop_words.append(word)
    return stop_words


def text_feature(train_data_list,test_data_list,feature_word):

    def text_feature(data,feature_word):
        data_words = set(data)
        feature = [1 if word in data_words else 0 for word in feature_word]
        # print "a"
        # print feature
        return feature

    train_feature = [text_feature(data,feature_word) for data in train_data_list]
    test_feature = [text_feature(data,feature_word) for data in test_data_list]
    # print train_feature,test_feature
    return train_feature,test_feature

def chi_feature(data_feature,class_list,feature_word):
    chi_model = SelectKBest(chi2,k=900)
    chi_feature = chi_model.fit_transform(data_feature,class_list)
    print chi_feature.shape
    feature_name = [feature_word[i] for i in chi_model.get_support(indices=True)]
    print json.dumps(feature_name,encoding='utf-8',ensure_ascii=False)
    joblib.dump(chi_model,'./model/chi_model.pkl')
    return chi_feature



def text_classifier(train_feature,train_class_list,test_feature,test_class_list):

    clf = MultinomialNB()
    classifier = clf.fit(train_feature,train_class_list)
    joblib.dump(classifier,'./model/bool_classifier.pkl')
    train_score = cross_val_score(clf,train_feature,train_class_list,cv=10,scoring='accuracy')
    train_score = train_score .mean()
    print train_score
    test_predict = clf.predict(test_feature)
    test_accuracy = classifier.score(test_feature,test_class_list)

    print "准确率", test_accuracy
    print"每个类别的精确率和召回率："
    print  classification_report(test_class_list, test_predict)

    return test_accuracy,train_score



if __name__ == '__main__':
    reload(sys)

    sys.setdefaultencoding('utf8')

    train_file = './cnews/cnews.test.txt'
    test_file = './cnews/cnews.val.txt'
    stopword_path = './cnews/stopwords.txt'
    stopword_list = get_stopword((stopword_path))

    test_acc = []
    train_acc = []
    feature_num = range(0,1,1)

    train_data_list,train_class_list = text_process(train_file,stopword_list)
    test_data_list,test_class_list = text_process(test_file,stopword_list)
    word_list = word_frequence_dict(train_data_list)
    for num in feature_num:
        feature_word = get_feature_word(word_list, num, stopword_list)
        # print feature_word
        # for word in feature_word:
        #     print word.encode('utf-8')
        train_feature,test_feature = text_feature(train_data_list,test_data_list,feature_word)
        train_chi_feature = chi_feature(train_feature,train_class_list,feature_word)
        test_chi_feature = chi_feature(test_feature,test_class_list,feature_word)
        # train_feature,test_feature = text_feature(train_data_list,test_data_list,feature_word)
        # test_accuracy,train_score = text_classifier(train_feature,train_class_list,test_feature,test_class_list)
        test_accuracy,train_score = text_classifier(train_chi_feature,train_class_list,test_chi_feature,test_class_list)

        test_acc.append(test_accuracy)
        train_acc.append(train_score)

    plt.figure()
    plt.plot(feature_num, test_acc)
    plt.plot(feature_num, train_acc, color='red')
    # plt.plot(feature_num, accuracy, color='yellow')
    plt.title('Relationship of feature numbers and test_accuracy')
    plt.xlabel('dimension')
    plt.ylabel('test_accuracy')
    plt.legend()
    plt.show()
    plt.savefig('result.png')

    print "finished"
