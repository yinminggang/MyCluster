# -*- coding: utf-8 -*-

import os.path
from com.util.conn_db import ConnMysql
import jieba
import re
from com.train.train_vector_model import trainWord2Vector, trainDoc2Vector
import numpy as np
from com.util.vector_computation import averageVector, manyVectorDistance
from com.cluster.singlepass import SinglePassCluster
import time
import logging
from com.model.doc import Doc
from com.model.word import Word
import math

def removePunctuation(title):
    """
    这里有两种方法，我们使用第二种繁琐的，因为我需要把每个标点的地方换成空格，这样便于分词
    :param title:
    :return:
    """
    punctuation = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗#〞︰︱︳﹐､﹒
            ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
            々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    """
    第一种方法：使用lambda表达式和filter函数 过滤标点符号
    # lambda [arg1[, arg2, ... argN]]: expression
    # filter(function, sequence)：对sequence中的item依次执行function(item)，
    # 将执行结果为True的item组成一个List/String/Tuple（取决于sequence的类型）
    # 对str/unicode
    str_filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
    # 对list
    list_filterpunt = lambda l: list(filter(lambda x: x not in punct, l))
    """
    title = str(title)
    for ch in title:

        if ch in punctuation:
            title = title.replace(ch, " ")
    return title

def dealWeiboContent(title):
    #result, number = re.subn("[\\s*]", "", title)
    #print(result)
    result = str(title)
    index1 = result.find("http://t.cn/")
    if index1 > -1:
        result = result.replace(result[index1:index1+19], "").strip()
    index1 = result.find("转发了", 0, 6)
    index2 = result.find("的微博:", 4)
    if index1 > -1 and index2 > -1:
        if index1 + 30 <= index2 or index1 >= index2:
            return ""
        # 去掉转发了...的微博:  只要微博正文
        result = result[index2+4:]
        # 转发理由：***** 赞[4] 转发[4] 评论[1]
        index1 = result.find("转发理由:", 25)
        index2 = result.find("赞[", 30)
        if index1 > -1 and index2 > -1 and index1 < index2:
            result = result[0:index1]
        # [组图共4张]  原图  赞[2]  原文转发[1] 原文评论[1]  转发理由:
        index1 = result.find("[组图共", 15)
        index2 = result.find("转发理由:", 25)
        if index1 < 0:
            # 原图 赞[18587] 原文转发[3567] 原文评论[1991]转发理由:轉發微博 全文 原图 赞[103]
            # 秒拍视频?赞[29]?原文转发[6]?原文评论[2]转发理由:
            index1 = result.find("原图", 30)
            if index1 < 0:
                index1 = result.find("秒拍视频", 30)
                if index1 < 0:
                    index1 = result.find("赞[", 30)

        if index1 > -1 and index2 > -1 and index1 < index2:
            result = result.replace(result[index1:index2+5], "").strip()
    else:
        # 非转发的微博
        index1 = result.find("赞[")
        index2 = result.find("原图")
        if index1 > -1 and index2 > -1 and index2 + 3 == index1:
            result = result[0: index2].strip()
        elif index1 != -1:
            result = result[0: index1].strip()
        elif index2 != -1:
            result = result[0: index2].strip()

        result, number = re.subn("[\\s*]", "", result)

    if result.endswith("原图"):
        result = result[0: len(result) - 2]
    index1 = result.find("组图共")
    if index1 > -1:
        result = result[0: index1-1].strip()

    result = result.replace("分享网易新闻","")
    result = result.replace("分享新浪新闻", "")
    result = result.replace("分享腾讯新闻", "")
    result = result.replace("分享搜狐新闻", "")
    result = result.replace("来自@网易新闻客户端", "")
    result = result.replace("来自@新浪新闻客户端", "")
    result = result.replace("来自@搜狐新闻客户端", "")
    result = result.replace("来自@腾讯新闻客户端", "")
    result = result.replace("来自@腾讯新闻客户端", "")
    result = result.replace("分享自凤凰新闻客户端", "")
    result = result.replace("分享自@凤凰视频客户端", "")
    result = result.replace("分享自@凤凰新闻客户端", "")
    result = result.replace("好文分享", "")
    result = result.replace("显示地图", "")
    result = result.replace("阅读全文请戳右边", "")
    result = result.replace("下载地址", "")
    result = result.replace("我在看新闻", "")

    return result.strip()

def removeExpression(title):
    """
    移除微博文本中的标签符号
    :return:
    """
    new_title = str(title)
    for ex in expression_list:
        if "[" in new_title and "]" in new_title:
            # print("存在[ and ]  %s " % ex)
            if ex[0] in new_title:
                # print("%s出现" % ex)
                new_title = new_title.replace(ex[0], "")
        else:
            break
    return new_title

def loadDataAndSave(start_time, end_time):
    # 加载过滤词
    filter_read = open("sources/filter_words.dic", mode='r', encoding='utf-8')
    filter_words = set()
    write_flag = 0
    for words in filter_read:
        words = words.strip("\n")
        if words in filter_words:
            write_flag = 1
            logging.info("过滤词典中有重复词:%s" % words)
        filter_words.add(words)
    filter_read.close()
    # 出现了重复词 则要更新词表
    if write_flag == 1:
        filter_write = open("sources/filter_words.dic", mode='w+', encoding='utf-8')
        for word in filter_words:
            filter_write.write(word+"\n")
        filter_write.close()

    conn = ConnMysql("59.77.233.198", 3306, "root", "mysql_fzu_118", "hottopicDatabase")
    conn.connectMysql()
    sql = "select distinct id,title from tdaily where day between " \
          "'"+start_time+"' and '"+end_time+"' order by day,id limit 150000"
    logging.warning(sql)
    all_data = conn.queryData(sql)
    # 存储微博中的表情符号（从数据库获取）
    global expression_list
    expression_list = conn.queryData("select expression from Expression")

    all_docs_list = []  # 存储文档类的集合
    all_words_list = dict()  # 存储所有词的字典<词，词类Word>
    sentence_write = open("sources/init_sentences.txt", mode='w+', encoding='utf-8')
    words_write = open("sources/splited_words.txt", mode='w+', encoding='utf-8')
    ids_write = open("sources/ids.txt", mode='w+', encoding='utf-8')
    jieba.load_userdict("sources/jieba_filter_dict.txt")
    index = 1
    word_num = 0
    all_nums = 0
    # 对于每篇微博
    for row in all_data:
        ids_write.write(str(row[0])+"\n")
        title = row[1]
        title = removeExpression(title)
        title = dealWeiboContent(title)
        sentence_write.write(title+'\n')
        title = removePunctuation(title)
        if len(title) < 6:
            continue
        words_tf = dict()
        seg_list = jieba.cut(title, cut_all=False)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        words = ""
        word_flag = 0
        splited_words = []
        for seg in seg_list:
            seg = seg.strip()
            if seg in filter_words or len(seg) < 2:
                continue
            splited_words.append(seg)
        if len(splited_words) < 3:
            continue
        for one_word in splited_words:
            words += " " + one_word
            all_nums += 1
            # 词类的更新
            if one_word in all_words_list.keys():
                temp_word = all_words_list[one_word]
                if word_flag == 0:
                    temp_word.__setattr__("doc_nums", temp_word.__getattribute__("doc_nums") + 1)
                    word_flag = 1
                temp_word.__setattr__("nums", temp_word.__getattribute__("nums") + 1)
                # all_words_list[one_word] = temp_word   # 可以不需要
            else:
                word_num += 1
                new_word = Word(one_word, 1, 1)
                all_words_list[one_word] = new_word

            all_words_list[one_word].__getattribute__("doc_index_list").append(index)
            # 文档类的更新
            if one_word in words_tf.keys():
                words_tf[one_word] += 1
            else:
                words_tf[one_word] = 1
        words = words.strip() + "\n"
        words_write.write(words)
        doc = Doc(row[0], index, title, splited_words=splited_words, words_tf=words_tf)
        all_docs_list.append(doc)
        del doc
        index += 1
    sentence_write.close()
    words_write.close()
    ids_write.close()
    conn.closeMysql()
    logging.info("词个数为：%s" % word_num)
    return all_nums, all_words_list, all_docs_list

def sentenceFeature(all_nums, all_docs_list, all_words_list):
    """
    开始构造每篇文章的特征向量，=每个文章中含有的词的特征向量的平均值
    先读word2vec.vector词特征向量，读成一个字典类型，<词，向量>
    """
    # 迭代标志
    iter_flag = 1
    while iter_flag > 0:
        iter_flag = -1
        ids_list = []
        titles_list = []
        word2vec = {}
        word2vec_read = open("result/pre_word2vec.vector", mode='r', encoding='utf-8')
        all_lines = word2vec_read.readlines()
        Ht = 0
        for line in all_lines[1:]:
            word = line.split(" ")
            # 将['0.01','0.02','0.03']-->[0.01,0.02,0.03]
            word2vec[word[0]] = list(map(float, word[1:]))
            temp_ht = all_words_list[word[0]].__getattribute__("nums")/all_nums
            Ht += temp_ht * math.log(temp_ht)
        if len(word2vec) < 1:
            logging.info("特征矩阵大小为0")
            return None
        word2vec_read.close()
        sentences_matrix = []
        index = 0
        feature_write = open("result/word2vec_sentence_features.vector", mode='w+', encoding='utf-8')
        # 重写splited_words 可能要后面要训练最新词向量
        words_write = open("sources/splited_words.txt", mode='w+', encoding='utf-8')
        while index < len(all_docs_list):
            words_matrix = []
            one_doc = all_docs_list[index]
            words = one_doc.__getattribute__("splited_words")
            doc_words_weight = []
            Htd = 0
            word_index = 0
            # 得出各个词的特征向量  并形成一个矩阵  然后计算平均值  就得到该句子的特征向量
            while word_index < len(words):
                word = words[word_index]
                if word in word2vec:
                    # 设置词的特征向量
                    all_words_list[word].__setattr__("word_matrix", word2vec[word])
                    # 计算每个词对每个句子的权重，参照2012年软件学报论文
                    LTij = math.log(one_doc.__getattribute__("words_tf")[word]+1)/(1+math.log(len(one_doc.__getattribute__("title"))))
                    # 求每个词的GT值
                    gt = 0
                    for doc_index in all_words_list[word].__getattribute__("doc_index_list"):
                        try:
                            temp_gt = all_docs_list[doc_index-1].__getattribute__("words_tf")[word]/all_words_list[word].__getattribute__("nums")
                        except Exception:
                            logging.error("出现错误")
                            logging.error(all_docs_list[doc_index - 1].__getattribute__("words_tf"))
                            logging.error(all_docs_list[doc_index - 1].__getattribute__("title"))
                            logging.error(word)
                            logging.error(all_words_list[word].__getattribute__("doc_index_list"))
                        gt += temp_gt * math.log(temp_gt)
                    GTi = 1+gt/math.log(len(all_docs_list))
                    temp_Htd = one_doc.__getattribute__("words_tf")[word]/len(one_doc.__getattribute__("title"))
                    Htd += temp_Htd * math.log(temp_Htd)
                    doc_words_weight.append(LTij*GTi)
                    words_matrix.append(np.array(word2vec[word]))
                else:
                    # print("不存在的词语%s==" % word)
                    # 词字典中删除该词 pop指定键值删除
                    all_words_list.pop(word)
                    one_doc.__setattr__("title", one_doc.__getattribute__("title").replace(word, "", num=50))
                    one_doc.__getattribute__("words_tf").pop(word)
                    words.remove(word)
                    word_index -= 1

            # 一个句子分析结束，开始计算变量值
            GDj = 1 - Htd/Ht
            # 列表中各个元素都乘以GDj  如：[1,2,3]-->[2,4,6]
            doc_words_weight = [x * GDj for x in doc_words_weight]
            words_write.write(' '.join(words) + "\n")
            one_doc.__setattr__("splited_words", words)
            ids_list.append(one_doc.__getattribute__("doc_index"))
            titles_list.append('||'.join(one_doc.__getattribute__("splited_words")))
            # titles_list.append(one_doc.__getattribute__("title"))
            # print(words)
            # print(doc_words_weight)
            # 说明该句子中的词全部没出现（被过滤掉了） 可直接作为垃圾文章，，则需要移除相应的id 和词
            if len(words_matrix) < 1:
                logging.error("移除句子相关词语为:%s" % one_doc.__getattribute__("splited_words"))
                logging.error(one_doc.__getattribute__("title"))
                logging.error(one_doc.__getattribute__("doc_index"))
                logging.error(one_doc.__getattribute__("doc_id"))
                all_docs_list.pop(index)
                index -= 1
                # 出现移除的 则需要再迭代，重新训练 再求特征向量
                iter_flag = 1
                continue
            if iter_flag < 0:
                # 只要出现了迭代标志为1，则就不写入，减少IO，因为后面还是要重新迭代求解，重新写入
                feature = averageVector(many_vectors=words_matrix, vector_weight=doc_words_weight, column_num=vector_dimension)
                feature_write.write(' '.join(str(f) for f in feature) + "\n")
                one_doc.__setattr__("feature_vector", feature)
                # 计算句子特征向量 并加入到矩阵中
                sentences_matrix.append(feature)
            index += 1
        feature_write.close()
        words_write.close()
        if iter_flag > 0:
            logging.info("再次训练")
            # 重新训练
            trainWord2Vector(sentence_count=len(all_docs_list), vector_dimension=vector_dimension, train_count=30)

    # 退出迭代过程
    # 重新更新id ,sentences和splited_words
    ids_write = open("sources/ids.txt", mode='w+', encoding='utf-8')
    sentences_write = open("sources/new_sentences.txt", mode='w+', encoding='utf-8')
    for doc in all_docs_list:
        ids_write.write(str(doc.__getattribute__("doc_index"))+"\n")
        sentences_write.write(doc.__getattribute__("title")+"\n")
    sentences_write.close()
    ids_write.close()

    # 返回新句子特征向量矩阵
    return ids_list, titles_list, sentences_matrix

def readDocFeature():
    sentence_matrix = []
    feature_read = open("result/doc2vec.vector", mode='r', encoding='utf-8')
    lines = feature_read.readlines()
    for line in lines:
        line = line.strip("\n")
        word = line.split(" ")
        # 将['0.01','0.02','0.03']-->[0.01,0.02,0.03]
        sentence_matrix.append(list(map(float, word)))
    feature_read.close()
    return sentence_matrix

def threaholdRange(feature_list):
    distance = manyVectorDistance(vec_a=feature_list[0], vec_b=feature_list[1], distance_type="Euclidean")
    min = distance
    max = distance
    dis = []
    for i in range(0, len(feature_list)-1, 1):
        for j in range(i+1, len(feature_list), 1):
            distance = manyVectorDistance(vec_a=feature_list[i], vec_b=feature_list[j], distance_type="Euclidean")
            if distance == 0:
                continue
            if distance < min:
                min = distance
                continue
            if distance > max:
                max = distance
                continue
    dis.append(min)
    dis.append(max)
    return dis

# 向量维度
vector_dimension = 50
train_type = "word"
if __name__ == '__main__':
    start_time = '2016-06-22 00:00:00'
    end_time = '2016-07-05 23:59:59'
    log_name = "logs/"+start_time[8:10]+"-"+end_time[8:10]+".txt"
    logging.basicConfig(filename=log_name, format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    stime = time.time()
    all_nums, all_words_list, all_docs_list = loadDataAndSave(start_time=start_time, end_time=end_time)
    length = len(all_docs_list)
    logging.info("first docs size is %s" % length)
    logging.info("first words size is %s" % len(all_words_list))
    if train_type == "word":
        trainWord2Vector(sentence_count=length, vector_dimension=vector_dimension, train_count=30)
        ids, titles, sentence_features = sentenceFeature(all_nums=all_nums, all_words_list=all_words_list, all_docs_list=all_docs_list)
    elif train_type == "doc":
        trainDoc2Vector(sentence_count=length, vector_dimension=vector_dimension, train_count=1)
        sentence_features = readDocFeature()
        logging.info("sentence_features size is %s" % len(sentence_features))
    length = len(all_docs_list)
    logging.info("new_docs size is %s" % length)
    logging.info("new docs size is %s" % length)
    if sentence_features is None:
        logging.info("is none")
        exit(0)
    max_index = int(0.01* length)
    logging.info("最大下标为：%s " % max_index)
    distance = threaholdRange(np.array(sentence_features[0:max_index]))
    logging.info("max and min distance is %s,%s" % (distance[0], distance[1]))
    logging.warning("start single pass cluster ")
    logging.info("the real docs is %s " % len(sentence_features))
    logging.info("the real docs is %s " % len(ids))
    clusted = SinglePassCluster(threshold_list=distance, ids_list=ids, vector_list=sentence_features, title_list=titles)
    # clusted.printClusterResult()
    logging.warning("end cluster and save clusters")
    clusted.saveClusterResult()
    logging.info("save successful")
    logging.info("总花费时间为：%.9fs seconds" % ((time.time()-stime)/1000))

