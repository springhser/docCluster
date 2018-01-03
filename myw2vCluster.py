#!/usr/bin/env python3
"""
Word2Vec Centroid Tool

Author: Gaetano Rossiello, University of Bari, Italy
Email: gaetano.rossiello@uniba.it

"""

import argparse
import logging
import time
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import TaggedLineDocument


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        wfile_no_p = open(self.filename, "r", encoding="utf-8")
        sentences = []
        titles = []
        for line in wfile_no_p.readlines():
            line = line.strip().split()
            yield TaggedDocument(words=line[1:], labels=['%s' % line[0]])

def d2ctest():
    documents = TaggedLineDocument("new_text2.txt")
    model = Doc2Vec(documents, size=10, window=2, min_count=1, workers=1)
    print(model)
    model


def w2vTest(filename):
    start = time.time()
    print("加载 w2v model ... ", end="", flush=True)
    sentences1 = word2vec.Text8Corpus("new_text2.txt")
    wfile_no_p = open("new_text1.txt", "r", encoding="utf-8")
    sentences = []
    titles = []
    for line in wfile_no_p.readlines():
        line =  line.strip().split()
        title = [line[0], len(line)-1]
        titles.append(title)
        sentences.append(line[1:])

    word_dict = {}
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    print(word_dict)
    print(len(word_dict))
    print(sentences)
    print(len(words))
    print(titles)
    print(len(titles))
    w2v_model = word2vec.Word2Vec(sentences, min_count=0, size=20)
    print("加载时间 {:.2f} 秒.".format(time.time() - start), flush=True)
    print(type(w2v_model))
    print(w2v_model)
    w2v_model.save("w2vModel")
    word_vectors = w2v_model.wv.syn0
    word_index = w2v_model.wv.index2word
    print(len(word_index))
    # estimator = PCA(n_components=2)
    # w2v_pca = estimator.fit_transform(word_vectors)
    # print(type(w2v_pca))

    print(type(word_vectors))
    print(word_index)
    # 降维并画图
    # plt.figure(figsize=(20, 20))
    # for i in range(len(word_index)):
    #     # print(word_index[i] + " " + str((w2v_pca[i][0]+1)*10)+","+str((w2v_pca[i][1]+1)*10))
    #     # plt.plot((w2v_pca[i][0]+1)*10,(w2v_pca[i][1]+1)*10, color='b', marker=word_index[i])
    #     x = (w2v_pca[i][0]+1)*10
    #     y = (w2v_pca[i][1]+1)*10
    #     plt.scatter(x, y)
    #     plt.annotate(word_index[i],
    #                  xy=(x, y),
    #                  xytext=(5, 2),
    #                  textcoords='offset points',
    #                  ha='right',
    #                  va='bottom')
    # plt.savefig("w2vc.png")
    #
    print("聚类开始 ... ", end="", flush=True)
    kmeans = KMeans(n_clusters=10, n_jobs=-1, random_state=0)#分为10类
    idx = kmeans.fit_predict(word_vectors)
    print("结束时间 {:.2f} sec.".format(time.time() - start), flush=True)

    start = time.time()
    print("输出文件... ", end="", flush=True)
    colors= ['b']
    word_centroid_list = list(zip(w2v_model.wv.index2word, idx))
    word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
    file_out = open("w2ccluster_res1.txt", "w", encoding="utf-8")
    file_out.write("词语\t类型号\n")
    for word_centroid in word_centroid_list_sort:
        line = word_centroid[0] + "\t" + str(word_centroid[1]) + '  \n'
        file_out.write(line)
    file_out.close()
    print("结束时间 {:.2f} sec.".format(time.time() - start), flush=True)
    file_out2 = open("w2ccluster_res2.txt", "w", encoding="utf-8")
    file_out2.write("被试编号 词语 \t类型号\t 词频 \n ")
    i = 0
    for sentence in sentences:
        new_line = titles[i][0] + ":  "
        for word in sentence:
            word_type = ""
            word_num = ""
            for word_centroid in word_centroid_list:
                if word == word_centroid[0]:
                    word_type = "type: "+str(word_centroid[1])
            if word in word_dict:
                word_num = "num: " + str(word_dict[word])
            new_line = new_line + "  "+ word + " " + word_type + " " + word_num + ','
        new_line = new_line + "\n"
        file_out2.write(new_line)
        i+=1
    file_out2.close()
    type_dict = {}
    type_word_dict = {}
    for word_centroid in word_centroid_list:
        if word_centroid[1] in type_dict:
            type_dict[word_centroid[1]] += 1
            type_word_dict[word_centroid[1]] += word_dict[word_centroid[0]]
        else:
            type_dict[word_centroid[1]] = 1
            type_word_dict[word_centroid[1]] = word_dict[word_centroid[0]]
    file_out3 = open("w2ccluster_res3.txt", "w", encoding="utf-8")
    file_out3.write("类型号\t 数量 \t 词频 \n ")
    for key in type_dict:
        new_wt = "Type: "+ str(key) + ", num: " + str(type_dict[key]) + "  " + "word_num: " + str(type_word_dict[key]) + "\n"
        file_out3.write(new_wt)
    file_out3.close()
    return

    # d2ctest()

