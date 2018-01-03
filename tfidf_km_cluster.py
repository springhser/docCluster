# coding=utf-8
"""
Created on 2016-01-06 @author: Eastmount
"""

import time
import re
import os
import sys
import codecs
import shutil
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":

    #########################################################################
    #                           第一步 计算TFIDF
    # 文档预料 空格连接
    corpus = []

    rfile = open('new_text1.txt', 'r', encoding="utf-8")
    alllist = [] # 存储每一个文档
    pairlist = []
    # 读取预料 一行预料为一个文档
    for line in rfile.readlines():
        if line is not None:
            print(line)
            line = line.strip()
            anslist = line.split()
            # 去掉每一个文档开头的被试编号
            pairlist.append(anslist[0])
            corpus.append(line[1:])
        # print corpus
    # time.sleep(1)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # resName = "output_result.txt"
    # result = codecs.open(resName, 'w', 'utf-8')
    # for j in range(len(word)):
    #     result.write(word[j] + ' ')
    # result.write('\r\n\r\n')
    #
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')
    # result.close()

    ########################################################################
    #                               第二步 聚类Kmeans

    print('Start Kmeans:')

    from sklearn.cluster import KMeans

    clf = KMeans(n_clusters=5)
    s = clf.fit(weight)
    print(s)

    # 20个中心点
    print(clf.cluster_centers_)

    cluster_result = "tfidf_cluster_res1.txt"
    wfile = open(cluster_result, 'w', encoding="utf-8")
    # 每个样本所属的簇
    print(clf.labels_)
    cluster_list = []
    atemp = []
    i = 0
    for i in range(len(clf.labels_)):
        print(pairlist[i])
        str1 = pairlist[i]
        new_line = "People: " + str(str1) + " " + " Type: " + str(clf.labels_[i]) + " \n"
        wfile.write(new_line)
        i += 1

    wfile.close()

    cluster_result1 = "tfidf_cluster_res2.txt"
    wfile1 = open(cluster_result1, 'w', encoding="utf-8")
    type_dict = {}
    for i in clf.labels_:
        if i in type_dict:
            type_dict[i] += 1
        else:
            type_dict[i] = 1

    for key in type_dict:
        type_line = "Type: " + str(key) + " " + "num: " + str(type_dict[key])  + "\n"
        wfile1.write(type_line)
    wfile1.close()
        # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)



