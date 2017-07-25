# MyCluster
word2vec for single-pass cluster

data: sina weibo
使用word2vec和doc2vec分别计算出句子向量，其中word2vec计算词向量，取取平均值作为句子向量，
根据句子向量余弦距离/欧式距离，作为相似度，最后进行聚类。
