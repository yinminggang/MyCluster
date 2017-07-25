# -*- coding: utf-8 -*-

import logging
import time

import numpy as np

from com.model import cluster_unit
from com.util.vector_computation import manyVectorDistance


class SinglePassCluster:
    def __init__(self, threshold_list=None, vector_list=None, ids_list=None, title_list=None):
        """
        :param t:一趟聚类的阈值
        :param vector_list:
        """
        self.threshold_list = threshold_list  # 一趟聚类的阈值
        self.threshold = 0.3
        self.vector_list = np.array(vector_list) # 存储所有文章的特征向量
        self.id_list = ids_list  # 存储所有文章的id,与特征向量相对应
        self.title_list = title_list  # 存储所有文章的内容
        self.cluster_list = []  # 聚类后簇的列表
        t1 = time.time()
        self.clustering()
        t2 = time.time()
        self.cluster_num = len(self.cluster_list)  # 聚类完成后  簇的个数
        self.spend_time = t2-t1  # 一趟聚类花费的时间

    def clustering(self):
        # 初始新建一个簇
        self.cluster_list.append(cluster_unit.ClusterUnit())
        # 读入的第一个文章（结点）归入第一个簇中
        self.cluster_list[0].addNode(node=self.id_list[0], node_vec=self.vector_list[0], title=self.title_list[0])
        length = len(self.id_list)
        # 遍历所有的文章  开始进行聚类  index 从1->(len-1)
        for index in range(length)[1:]:
            if self.threshold_list is not None:
                if self.threshold < (self.threshold_list[1]+self.threshold_list[0]) * 0.5:
                    self.threshold = index*(self.threshold_list[1] - self.threshold_list[0])/100
                    logging.info("threshold is %s" % self.threshold)
                else:
                    self.threshold = (self.threshold_list[1] + self.threshold_list[0]) * 0.3
                    logging.error("the last threshold is %s" % self.threshold)

            current_vector = self.vector_list[index]
            if current_vector is None:
                logging.info("index=%s" % index)
                logging.info("len(vectors)=%s" % len(self.vector_list))
            # 与簇的质心的最小距离
            min_distance = manyVectorDistance(distance_type="Euclidean", vec_a=current_vector,
                                              vec_b=self.cluster_list[0].centroid)
            # 最小距离的簇的索引
            min_cluster_index = 0
            for cluster_index, one_cluster in enumerate(self.cluster_list[1:]):
                # enumerate会将数组或列表组成一个索引序列
                # 寻找距离最小的簇，记录下距离和对应的簇的索引
                distance = manyVectorDistance(distance_type="Euclidean", vec_a=current_vector,
                                              vec_b=one_cluster.centroid)
                try:
                    if distance < min_distance:
                        min_distance = distance
                        # 因为cluster_index是从0开始
                        min_cluster_index = cluster_index + 1
                except TypeError:
                    logging.info(distance)

            logging.info("min_distance is %s " % min_distance)
            # 最小距离小于阈值，则归于该簇
            if min_distance < self.threshold:
                self.cluster_list[min_cluster_index].addNode(node=self.id_list[index], node_vec=current_vector,
                                                             title=self.title_list[index])
            else:
                new_cluster = cluster_unit.ClusterUnit()
                new_cluster.addNode(node=self.id_list[index], node_vec=current_vector, title=self.title_list[index])
                self.cluster_list.append(new_cluster)
                del new_cluster

    def printClusterResult(self, label_dict=None):
        # 打印出聚类结果
        # label_dict:节点对应的标签字典
        logging.info("**********single-pass cluster result******")
        for index, one_cluster in enumerate(self.cluster_list):
            # print("cluster_index:%s" % index)
            # 簇的结点列表
            # print(one_cluster.node_list)
            if label_dict is not None:
                # 若有提供标签字典，则输出该簇的标签
                logging.info(" ".join([label_dict[n] for n in one_cluster.node_list]))
                logging.info("node num:%s" % one_cluster.node_num)
                logging.info("========================")
        logging.info("the number of nodes %s" % len(self.vector_list))
        logging.info("the number of cluster %s" % self.cluster_num)
        logging.info("spend time %.9fs" % (self.spend_time / 1000))

    def saveClusterResult(self):
        """
        把各个聚类结果写到各个文件中
        :return: 
        """
        logging.info("the number of cluster %s" % self.cluster_num)
        logging.info("spend time of cluster %.9fs" % (self.spend_time / 1000))
        for index, one_cluster in enumerate(self.cluster_list):
            if one_cluster.node_num < 7:
                continue
            if one_cluster.node_num < 3:
                cluster_write = open("result/22cluster%s.txt" % index, mode='w+', encoding='utf-8')
            elif one_cluster.node_num < 4:
                cluster_write = open("result/33cluster%s.txt" % index, mode='w+', encoding='utf-8')
            else:
                cluster_write = open("result/cluster%s.txt" % index, mode='w+', encoding='utf-8')
            cluster_write.write("类别%s情况如下：\n" % index)
            cluster_write.write("共有%s篇文章\n" % one_cluster.node_num)
            cluster_write.write("类别质心向量如下:\n%s" % one_cluster.centroid)
            cluster_write.write("\n文章id和内容如下:\n")
            for i, id in enumerate(one_cluster.node_list):
                cluster_write.write("%s %s\n" % (id, one_cluster.title_list[i]))

            cluster_write.close()
