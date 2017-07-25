# -*- coding: utf-8 -*-

class Word:
    """
    表示词语类，也就是每个词
    """
    word = ""  # 词内容
    doc_nums = 0  # 包含该词的文档个数
    nums = 0  # 整个文档集中词的出现次数
    def __init__(self, word, doc_nums, nums):
        self.doc_index_list = []  # 包含该词的文档下标list
        self.word_matrix = []  # word2vec计算出来的词向量
        self.word = word
        self.doc_nums = doc_nums
        self.nums = nums
