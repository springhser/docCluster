#!/usr/bin/env python
# encoding: utf-8
"""
@version: 0.1
@author: springhser
@license: Apache Licence 
@contact: endoffight@gmail.com
@site: http://www.springhser.com
@software: PyCharm Community Edition
@file: lsa_text_pro.py
@time: 2017/8/2 22:29
"""
from gensim import corpora, models
texts = []

rfile = open('new_text2.txt', 'r', encoding="utf-8")
alllist = [] # 存储每一个文档
# pairlist = []
# # 读取预料 一行预料为一个文档
for line in rfile.readlines():
    if line is not None:
        anslist = line.split()
        texts.append(anslist)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)
dictionary.save("textdict_lsa.dict")
corpora.MmCorpus.serialize("corporadict_lsa.mm", corpus)




