# -*- coding: utf-8 -*-

from com.cluster.cluster_weather import SinglePassCluster
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as np
from com.util.vector_computation import averageVector

if __name__ == '__main__':
    read = open('1.txt', mode='r', encoding='utf-8')
    write = open('splited.txt', mode='w+', encoding='utf-8')
    lines = read.readlines()
    splited_words = []
    sentences = []
    for line in lines:
        line = line.strip("\n")
        sentences.append(line)
        seg_list = jieba.cut(line, cut_all=False)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        words = ""
        for seg in seg_list:
            words += " " + seg
        write.write(words+"\n")
        splited_words.append(words)

    read.close()
    write.close()
    inp, outp1, outp2 = "splited.txt", "word2vec.model", "pre_word2vec.vector"
    sentence = LineSentence(inp)
    model = Word2Vec(sentence, sg=1, size=50, window=8, min_count=0, workers=multiprocessing.cpu_count())
    for i in range(10):
        model.train(sentences=sentence, total_examples=32859, epochs=model.iter)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    word2vec = {}
    word2vec_read = open("pre_word2vec.vector", mode='r', encoding='utf-8')
    all_lines = word2vec_read.readlines()
    for line in all_lines[1:]:
        word = line.split(" ")
        # 将['0.01','0.02','0.03']-->[0.01,0.02,0.03]
        word2vec[word[0]] = list(map(float, word[1:]))
    if len(word2vec) < 1:
        print("特征矩阵大小为0")
    word2vec_read.close()
    sentences_matrix = []
    index = 0
    feature_write = open("sentence_features.vector", mode='w+', encoding='utf-8')
    while index < len(splited_words):
        words_matrix = []
        words = splited_words[index].split(" ")
        # 得出各个词的特征向量  并形成一个矩阵  然后计算平均值  就得到该句子的特征向量
        for word in words:
            if word in word2vec:
                words_matrix.append(np.array(word2vec[word]))

        # 说明该句子中的词全部没出现（被过滤掉了） 可直接作为垃圾文章，，则需要移除相应的id 和词
        if len(words_matrix) < 1:
            print("相关词语为:%s" % splited_words[index])
            splited_words.pop(index)
            # 出现移除的 则需要再迭代，重新训练 再求特征向量
            continue
            # 只要出现了迭代标志为1，则就不写入，减少IO，因为后面还是要重新迭代求解，重新写入
        feature = averageVector(many_vectors=words_matrix, column_num=50)
        feature_write.write(' '.join(str(f) for f in feature) + "\n")
        # 计算句子特征向量 并加入到矩阵中
        sentences_matrix.append(feature)
        index += 1

    feature_write.close()
    clusted = SinglePassCluster(threshold=1.35, vector_list=sentences_matrix, content_list=sentences)
    clusted.printClusterResult()