# -*- coding: utf-8 -*-

import time

import matplotlib.pylab as pl
import numpy as np

from com.model import cluster_unit
from com.util.vector_computation import manyVectorDistance


class SinglePassCluster:
    def __init__(self, threshold, vector_list, content_list):
        """
        :param t:一趟聚类的阈值
        :param vector_list:
        """
        self.threshold = threshold  # 一趟聚类的阈值
        self.vectors = np.array(vector_list) #存储每篇文章的特征向量
        self.content_list = content_list
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
        self.cluster_list[0].addNode(0, self.vectors[0], self.content_list[0])
        # 遍历所有的文章  开始进行聚类  index 从1->(len-1)
        for index in range(len(self.vectors))[1:]:
            current_vector = self.vectors[index]
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
                if distance < min_distance:
                    min_distance = distance
                    # 因为cluster_index是从0开始，(个人觉得不用+1)
                    min_cluster_index = cluster_index + 1
            # 最小距离小于阈值，则归于该簇
            if min_distance < self.threshold:
                self.cluster_list[min_cluster_index].addNode(index, current_vector, self.content_list[index])
            else:
                new_cluster = cluster_unit.ClusterUnit()
                new_cluster.addNode(index, current_vector, self.content_list[index])
                self.cluster_list.append(new_cluster)
                del new_cluster

    def printClusterResult(self, label_dict=None):
        # 打印出聚类结果
        # label_dict:节点对应的标签字典
        print("**********single-pass cluster result******")
        for index, one_cluster in enumerate(self.cluster_list):
            print("cluster_index:%s" % index)
            #簇的结点列表
            print(one_cluster.node_list)
            print(one_cluster.title_list)
            if label_dict is not None:
                # 若有提供标签字典，则输出该簇的标签
                print(" ".join([label_dict[n] for n in one_cluster.node_list]))
                print("node num:%s" % one_cluster.node_num)
                print("========================")
        print("the number of nodes %s" % len(self.vectors))
        print("the number of cluster %s" % self.cluster_num)
        print("spend time %.9fs" % (self.spend_time / 1000))


if __name__ == '__main__':
    # 读取测试集 # 读取聚类特征
    temperature_all_city = np.loadtxt('../../sources/weather.txt', delimiter=",", usecols=(3, 4))
    print(type(temperature_all_city))
    #print(temperature_all_city)
    xy = np.loadtxt('../../sources/weather.txt', delimiter=",", usecols=(8, 9))  # 读取各地经纬度
    f = open('../../sources/weather.txt', 'r')
    lines = f.readlines()
    # 读取地区名称并转化为字典
    zone_dict = [i.split(',')[1] for i in lines]
    print("========")
    print(len(zone_dict))
    print(type(zone_dict))
    print(zone_dict[0])
    print(zone_dict)
    f.close()

    # 构建一趟聚类器
    clustering = SinglePassCluster(vector_list=temperature_all_city, threshold=9)
    clustering.print_result(label_dict=zone_dict)

    # 将聚类结果导出图
    fig, ax = pl.subplots()
    fig = zone_dict
    c_map = pl.get_cmap('jet', clustering.cluster_num)
    c = 0
    for cluster in clustering.cluster_list:
        for node in cluster.node_list:
            ax.scatter(xy[node][0], xy[node][1], c=c, s=30, cmap=c_map, vmin=0,
                       vmax=clustering.cluster_num)
        c += 1
    pl.show()