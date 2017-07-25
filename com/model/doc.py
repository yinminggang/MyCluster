# -*- coding: utf-8 -*-

class Doc:
    """
    表示文档类，也就是每篇微博
    """
    doc_id = ""  # 文档在表中的id
    doc_index = -1  # 文档的下标
    title = ""  # 微博内容
    splited_words = []  # 内容分词结果
    words_tf = dict()  # 每个词在文档中的出现次数<word,次数>

    def __init__(self, doc_id, doc_index, title, splited_words, words_tf):
        self.feature_vector = []  # 文档的句子特征向量
        self.doc_id = doc_id
        self.doc_index = doc_index
        self.title = title
        self.splited_words = splited_words
        self.words_tf = words_tf
