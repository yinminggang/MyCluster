# -*- coding: utf-8 -*-


from com.util.conn_db import ConnMysql
import jieba
import jieba.posseg as pseg
import logging
import re
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

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
    sql = "select t.id,t.title,t.day,t.createtime,t.author,t.uid,t.retransmit,t.replies,t.praisenum,t.initialUid," \
          "t.initialAuthor,fwi.createtime as createtime3,fwi.retransmit as retransmit3,fwi.replies as replies3," \
          "fwi.praisenum as praisenum3,wi.createtime as createtime1,wi.retransmit as retransmit1,wi.replies as " \
          "replies1,wi.praisenum as praisenum1,hwi.createtime as createtime2,hwi.retransmit as retransmit2," \
          "hwi.replies as replies2,hwi.praisenum as praisenum2 from ((tdaily t LEFT OUTER JOIN weiboinfoagain wi " \
          "on t.id=wi.id) LEFT OUTER JOIN home_weiboinfoagain hwi on t.id=hwi.id) LEFT OUTER JOIN first_weiboinfoagain" \
          " fwi on t.id=fwi.id where t.createTime>=t.day and t.day >='"+start_time+"' and t.field_id=3 and " \
          "t.day < '"+end_time+"' order by t.day asc,t.id asc "

    all_data = conn.queryData(sql)
    # 存储微博中的表情符号（从数据库获取）
    global expression_list
    expression_list = conn.queryData("select expression from Expression")
    ids = []
    splited_words = []
    words_write = open("com/com/test1/test1sources/splited_words.txt", mode='w+', encoding='utf-8')
    jieba.load_userdict("sources/jieba_filter_dict.txt")
    word_attribute = ['n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng', 'v', 'vd', 'vn',
                      'vf', 'vx', 'vi', 'vl', 'vg', 'a', 'an', 'ad', 'ag', 'al', 'mg']
    # 对于每篇微博
    for row in all_data:
        title = row[1]
        title = removeExpression(title)
        title = dealWeiboContent(title)
        #title = removePunctuation(title)
        seg_list = pseg.cut(title)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        words = ""
        for seg in seg_list:
            if seg.flag not in word_attribute:
                continue
            w = seg.word.strip()
            if len(w) < 1 : continue
            words += " " + w
        words = words.strip()
        if len(words) < 1:
            continue
        ids.append(row[0])
        splited_words.append(words)
        words = words + "\n"
        words_write.write(words)
    words_write.close()
    conn.closeMysql()
    print("句子大小为%s==%s" % (len(ids), len(splited_words)))
    return ids, splited_words

def trainWord2Vector(sentence_count, vector_dimension, train_count):

    lines, model_out, vector_out = "com/com/test1/test1sources/splited_words.txt", \
                                   "com/com/test1/test1sources/word2vec.model", \
                                   "com/com/test1/test1sources/word2vec.vector"
    logging.info("开始训练数据")
    sentences = LineSentence(lines)
    # 注意min_count=3表示词频小于3的词 不做计算，，也不会保存到word2vec.vector中
    # workers是训练的进程数，一般等于CPU核数  默认是3
    model = Word2Vec(sentences, sg=1, size=vector_dimension, window=8,
                     min_count=0, workers=multiprocessing.cpu_count())
    # 多训练几次  使得效果更好
    for i in range(train_count):
        model.train(sentences=sentences, total_examples=sentence_count, epochs=model.iter)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_out)
    model.wv.save_word2vec_format(vector_out)

def sentenceFeature(ids, splited_words):
    """
    开始构造每篇文章的特征向量，=每个文章中含有的词的特征向量的平均值
    先读word2vec.vector词特征向量，读成一个字典类型，<词，向量>
    """
    # 迭代标记
    iter_flag = 1
    sentences_matrix = []
    iter_end = 0
    while iter_flag > 0:
        print("又一轮迭代")
        iter_flag = -1
        word2vec = {}
        word2vec_read = open("com/com/test1/test1sources/word2vec.vector", mode='r', encoding='utf-8')
        all_lines = word2vec_read.readlines()
        for line in all_lines[1:]:
            word = line.split(" ")
            # 将['0.01','0.02','0.03']-->[0.01,0.02,0.03]
            word2vec[word[0]] = list(map(float, word[1:]))
        if len(word2vec) < 1:
            print("特征矩阵大小为0")
            return None
        word2vec_read.close()
        print("词向量大小为：%s "%len(word2vec))
        sentences_matrix = []
        feature_write = open("com/com/test1/test1sources/word2vec_sentence_features.vector", mode='w+', encoding='utf-8')
        words_write = open("com/com/test1/test1sources/splited_words.txt", mode='w+', encoding='utf-8')
        index = 0
        while index < len(ids):
            words_matrix = []
            words = splited_words[index].split(" ")
            i = 0
            # 得出各个词的特征向量  并形成一个矩阵  然后计算平均值  就得到该句子的特征向量
            # 因为接下来的遍历后面有remove操作，所以遍历需要小心  remove后会跳过某个元素
            while i < len(words):
                if words[i] in word2vec.keys():
                    words_matrix.append(np.array(word2vec[words[i]]))
                else:
                    iter_flag = 1
                    print(words)
                    print("词语不存在%s" % words[i]+"===")
                    words.pop(i)
                    # 这个一定要的，，不然遍历出问题，，会跳过元素
                    i -= 1
                i += 1
            if len(words) > 0:
                ss = ' '.join(words)
                words_write.write(ss + "\n")
                splited_words[index] = ss

            # 说明该句子中的词全部没出现（被过滤掉了） 可直接作为垃圾文章，，则需要移除相应的id 和词
            if len(words_matrix) < 1:
                print("移除句子id为%s，相关词语为:%s" % (ids[index], splited_words[index]))
                ids.pop(index)
                splited_words.pop(index)
                index -= 1
                continue
            else:
                feature = averageVector(many_vectors=words_matrix, column_num=vector_dimension)
                feature_write.write(str(ids[index])+" "+' '.join(str(f) for f in feature)+"\n")
                # 计算句子特征向量 并加入到矩阵中
                sentences_matrix.append(feature)
            index += 1

        words_write.close()
        feature_write.close()
        logging.info("%s==%s" % (len(ids), len(splited_words)))
        # 这表示所有迭代结束
        if iter_end == 1:
            print("所有迭代结束")
            break
        if iter_flag > 0:
            logging.info("再次训练")
            trainWord2Vector(sentence_count=len(splited_words), vector_dimension=vector_dimension, train_count=1)
        else:
            # 最后再训练一次  更改训练次数,不过还需要再迭代一次
            print("开始最后一次训练")
            trainWord2Vector(sentence_count=len(splited_words), vector_dimension=vector_dimension, train_count=30)
            iter_end = 1
            iter_flag = 1

    # 重新更新id ,sentences
    ids_write = open("sources/new_ids.txt", mode='w+', encoding='utf-8')
    for line in ids:
        ids_write.write(str(line)+"\n")
    ids_write.close()
    # 返回新句子特征向量矩阵
    return sentences_matrix

def averageVector(many_vectors, column_num):
    """
    求多个向量的平均向量
    :param many_vector:
    :column_num:向量列数
    :return:
    """
    average_vector = []
    for i in range(0, column_num, 1):
        average_vector.append(0)
    row_num = len(many_vectors)

    # 先求出各个列之和  后面再求平均值
    for vector in many_vectors:
        for i in range(0, column_num, 1):
            average_vector[i] = average_vector[i] + float(vector[i])

    for i in range(0, column_num, 1):
        average_vector[i] = average_vector[i] / row_num

    # 返回list类型的平均向量  [0.002,0.003,....]
    return average_vector

vector_dimension=100
if __name__ == '__main__':
    start_time = '2016-06-22 00:00:00'
    end_time = '2016-07-05 23:59:59'
    log_name = "com/com/test1/test1sources/" + start_time[8:10] + "-" + end_time[8:10] + ".txt"
    logging.basicConfig(filename=log_name, format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    ids, splited_words = loadDataAndSave(start_time, end_time)
    trainWord2Vector(sentence_count=len(ids), vector_dimension=vector_dimension, train_count=1)
    sentenceFeature(ids=ids, splited_words=splited_words)