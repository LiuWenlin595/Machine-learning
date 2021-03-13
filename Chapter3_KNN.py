#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 20:30
# @Author  : Wenlin Liu
# @File    : Chapter3_KNN.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
k值选择: 25
距离度量: 欧氏距离
-------------------------
运行结果:
准确率: 96.09%
精确率: 96.21%
召回率: 96.06%
运行时间: 12655.63s
"""

from utils import *
import numpy as np
import time
from tqdm import *


class MyKNN:
    """实现线性搜索KNN模型

    算法步骤:
    1. 加载训练集, 无需训练, 设定k值
    2. 测试: 对测试集的样本, 计算其与训练集所有样本的距离, 从中选出k个最相近样本, 采用投票法确定测试样本的分类(line 53)
    3. 统计测试结果, 分析指标

    """
    def __init__(self, file, k=1):
        data, labels = load_data(file)
        self.data, self.labels = np.mat(data), np.mat(labels).T
        self.n_cls = 10
        self.n_sample = np.shape(self.data)[0]
        self.k = k

    def test(self, data, labels, n_test=200):
        """测试函数"""
        print("Start test!")
        n_sample = n_test
        n_real_list, n_pred_list, n_correct_list = np.zeros(self.n_cls), np.zeros(self.n_cls), np.zeros(self.n_cls)

        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            classify_vote = np.zeros(self.n_cls)   # 0-9数字, 十分类任务
            top_k = [[-1, 1e5]] * self.k
            # 线性遍历训练集搜索
            for j in range(self.n_sample):
                dist = cal_dist_2_norm(self.data[j], xi)
                # 能想到的最好的数据结构了..., 拒绝用np.argsort
                insert_index = self.k
                while insert_index > 0 and dist < top_k[insert_index-1][1]:
                    insert_index -= 1
                for k in range(self.k-1, insert_index, -1):
                    top_k[k] = top_k[k-1]
                if insert_index < self.k:
                    top_k[insert_index] = [self.labels[j], dist]
            # 预测以及统计
            for cls, dist in top_k:
                classify_vote[cls] += 1
            pred = np.argmax(classify_vote)
            n_real_list[yi] += 1
            n_pred_list[pred] += 1
            if yi == pred:
                n_correct_list[yi] += 1

        evaluate(n_correct_list, n_real_list, n_pred_list, n_sample)


if __name__ == '__main__':
    start = time.time()

    my_knn = MyKNN('./Mnist/mnist_train.csv', k=25)
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.mat(test_data), np.mat(test_labels).T

    my_knn.test(test_data, test_labels, n_test=np.shape(test_data)[0])

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
