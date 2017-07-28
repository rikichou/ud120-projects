#!/usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    ### pickle是用于存储python对象的一个库，所以都出来的也是一个对象，authors就是作者的列表authors = [1,1,1,1,0,0,0,.....]
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    ### 同上，word_data是存储的每一封邮件的内容的列表
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    ###
    ### fit_transform用于生成一个语料库（fit的作用），语料库中包含features_train所有已经出现的单词。transform会将你的输入的每一行（每一封邮件）的每个单词计算一个权重值，
    ### 该权重值是基于整个语料库而言（features_train就是语料库），在邮件中没有出现的单词按照0计算。
    features_train_transformed = vectorizer.fit_transform(features_train)
    ###
    ### 该句和上一句的区别在于没有fit，所以就不会再生成语料库了，而是使用上一句生成的语料库，所以输出的权重值是相对于之前的语料库
    features_test_transformed  = vectorizer.transform(features_test)

    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    ### 由于特征过于庞大（features_train_transformed的列数代表特征的数量，大概有37851）,选择排名前10%的特征,大概剩余3785(测试集和训练集)
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
