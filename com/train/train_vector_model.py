# -*- coding: utf-8 -*-

import multiprocessing
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedLineDocument
import logging

def trainWord2Vector(sentence_count, vector_dimension, train_count):

    lines, model_out, vector_out = "sources/splited_words.txt", "result/word2vec.model", "result/pre_word2vec.vector"
    logging.info("开始训练数据")
    sentences = LineSentence(lines)
    # 注意min_count=3表示词频小于3的词 不做计算，，也不会保存到word2vec.vector中
    # workers是训练的进程数，一般等于CPU核数  默认是3
    # sg表示选择的训练算法
    model = Word2Vec(sentences, sg=1, size=vector_dimension, window=8,
                     min_count=0, workers=multiprocessing.cpu_count())
    # 多训练几次  使得效果更好
    for i in range(train_count):
        model.train(sentences=sentences, total_examples=sentence_count, epochs=model.iter)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_out)
    model.wv.save_word2vec_format(vector_out)

def trainDoc2Vector(sentence_count, vector_dimension):
    # train and save the model
    sentences = TaggedLineDocument('sources/splited_words.txt')
    model = Doc2Vec(sentences, size=vector_dimension, window=8, min_count=2, workers=multiprocessing.cpu_count())
    model.train(sentences, total_examples=sentence_count, epochs=model.iter)
    model.save('result/doc2vec.model')
    # save vectors
    out = open('result/doc2vec.vector', mode='w+', encoding='utf-8')
    for index in range(0, sentence_count, 1):
        docvec = model.docvecs[index]
        out.write(' '.join(str(f) for f in docvec) + "\n")

    out.close()