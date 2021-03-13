#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 18:53
# @Author  : Wenlin Liu
# @File    : Chapter4_NaiveBayes.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果(极大似然估计 + 无log累乘):
准确率: 23.64%
精确率: 54.03%
召回率: 23.07%
运行时间: 86.40s

运行结果(最大后验估计 + log累加):
准确率: 83.02%
精确率: 83.45%
召回率: 82.56%
运行时间: 86.70s
"""

from utils import *
import numpy as np
import time
from tqdm import *


class MyNaiveBayes:
    """实现朴素贝叶斯模型

    算法步骤:
    1. 加载数据集, 设定类别数量
    2. 计算一些测试需要用到的公式, 采用极大似然估计+平滑系数, 用log和无log准确率差别很大, 可能是因为下溢出导致的(line 50)
    3. 测试: 根据朴素贝叶斯分类的基本公式计算样本属于各类别的概率, 通过取最大概率估计样本的类别(line 81)
    4. 统计测试结果, 分析指标

    """
    def __init__(self, file):
        self.data, self.labels = load_data(file)
        self.n_cls = 10
        self.n_sample, self.n_feature = np.shape(self.data)
        self.prob_y_list, self.n_cls_list = np.ones(self.n_cls), np.zeros(self.n_cls)   # 十分类任务
        self.prob_x_cond_y = np.ones((self.n_cls, self.n_feature, 256))
        self.logprob_y_list = np.zeros(self.n_cls)
        self.logprob_x_cond_y = np.zeros((self.n_cls, self.n_feature, 256))
        self.preprocess()

    def preprocess(self):
        """朴素贝叶斯算法的公式预处理

        prob_y_list: 类别的先验概率
        n_cls_list: 类别的个数
        prob_x_cond_y: 在已知类别的条件下特征取各个值的概率
                       维度1对应y的类别, 维度2对应x的类别, 维度3对应特征取值

        """
        print("Start preprocess!")
        for i in tqdm(range(self.n_sample)):
            xi, yi = self.data[i], self.labels[i]
            self.n_cls_list[yi] += 1
            for j, v in enumerate(xi):
                self.prob_x_cond_y[yi][j][v] += 1
        self.prob_y_list = np.copy(self.n_cls_list)
        self.prob_y_list /= (self.n_sample + self.n_cls)
        self.logprob_y_list = np.log(self.prob_y_list)
        for i in range(self.n_cls):
            self.prob_x_cond_y[i] /= (self.n_cls_list[i] + self.n_feature)
            self.logprob_x_cond_y[i] = np.log(self.prob_x_cond_y[i])

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_list, n_pred_list, n_correct_list = np.zeros(self.n_cls), np.zeros(self.n_cls), np.zeros(self.n_cls)

        for sample in tqdm(range(n_sample)):
            xi, yi = data[sample], labels[sample]
            prob_cls_list = np.zeros(self.n_cls)   # 每个类别的概率
            for i in range(self.n_cls):
                prob = self.logprob_y_list[i]
                for j, v in enumerate(xi):
                    prob += self.logprob_x_cond_y[i][j][v]
                prob_cls_list[i] = prob
            # 预测以及统计
            pred = np.argmax(prob_cls_list)
            n_real_list[yi] += 1
            n_pred_list[pred] += 1
            if yi == pred:
                n_correct_list[yi] += 1

        evaluate(n_correct_list, n_real_list, n_pred_list, n_sample)


if __name__ == "__main__":
    start = time.time()

    my_naivebayes = MyNaiveBayes('./Mnist/mnist_train.csv')
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')

    my_naivebayes.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
