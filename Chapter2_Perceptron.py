#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 16:17
# @Author  : Wenlin Liu
# @File    : Chapter2_Perceptron.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
迭代轮次: 30
-------------------------
运行结果(no batch):
准确率: 80.29%
精确率: 77.45%
召回率: 83.87%
运行时间: 69.09s

运行结果(batch_size=200):
准确率: 86.06%
精确率: 87.46%
召回率: 83.25%
运行时间: 56.70s
"""

from utils import *
import numpy as np
import time
from tqdm import *


class MyPerceptron:
    """实现原始形式的感知机模型, 由于二分类(<5 and >=5)导致同类样本的特征值比较杂, 实际会很影响分类效果
        可能是因为用了np.mat导致计算速度比较慢, logistic regression用np.array算起来就快很多

    算法步骤:
    1. 加载训练集, 初始化w b, 初始化学习率α
    2. 训练: 根据原始形式的损失函数进行梯度更新(line 69), 设定迭代次数
    3. 测试: 使用更新完的w b对测试集进行计算(line 81)
    4. 统计测试结果, 分析指标

    """
    def __init__(self, file, learning_rate=1e-4, batch_size=200):
        data, labels = load_data(file)
        self.data, self.labels = np.mat(data), np.mat(preprocess(labels)).T
        self.n_sample, self.n_feature = np.shape(self.data)
        self.w, self.b = np.zeros((1, self.n_feature)), 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, iter, use_batch=False):
        """训练函数"""
        print("Start train!")
        for _ in tqdm(range(iter)):
            if use_batch:
                for i in range(0, self.n_sample, self.batch_size):
                    total_grad_w, total_grad_b = np.zeros((1, self.n_feature)), 0
                    for j in range(i, i+self.batch_size):
                        xi, yi = self.data[j], self.labels[j]
                        if yi * (self.w * xi.T + self.b) <= 0:
                            total_grad_w += yi * xi
                            total_grad_b += yi
                    mean_grad_w = total_grad_w / self.batch_size
                    mean_grad_b = total_grad_b / self.batch_size
                    self.w = self.w + self.learning_rate * mean_grad_w
                    self.b = self.b + self.learning_rate * mean_grad_b
            else:
                for i in range(self.n_sample):
                    xi, yi = self.data[i], self.labels[i]
                    # 误分类, 梯度下降, 梯度是 -1 * yi * xi
                    if yi * (self.w * xi.T + self.b) <= 0:
                        self.w = self.w + self.learning_rate * yi * xi
                        self.b = self.b + self.learning_rate * yi

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_pos, n_pred_pos, n_correct_pos, n_correct = 0, 0, 0, 0
        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            pred = self.w * xi.T + self.b
            if pred > 0:
                n_pred_pos += 1
            if yi > 0:
                n_real_pos += 1
            if yi > 0 and pred > 0:
                n_correct_pos += 1
            if yi * pred > 0:
                n_correct += 1

        evaluate_binary_classify(n_correct, n_correct_pos, n_real_pos, n_pred_pos, n_sample)


def preprocess(labels):
    """将Mnist数据集转为二分类任务"""
    for i, v in enumerate(labels):
        # 这里把负样本设成过0, debug花了点时间
        labels[i] = -1 if v < 5 else 1
    return labels


if __name__ == '__main__':
    start = time.time()

    my_perceptron = MyPerceptron('./Mnist/mnist_train.csv')
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.mat(test_data), np.mat(preprocess(test_labels)).T

    my_perceptron.train(iter=30, use_batch=False)
    my_perceptron.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
