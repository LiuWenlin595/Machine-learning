#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 18:58
# @Author  : Wenlin Liu
# @File    : Chapter6_LogisticRegression.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果(no batch):
准确率: 86.30%
精确率: 84.74%
召回率: 87.60%
运行时间: 37.68s

运行结果(batch_size=200, 学习率太低且更新次数太少导致有点欠拟合):
准确率: 76.73%
精确率: 74.34%
召回率: 79.61%
运行时间: 37.27s
"""
from utils import *
import numpy as np
import time
from tqdm import *


class MyLogisticRegression:
    """实现logistic regression模型, 对二分类问题给予概率输出, 缓解感知机模型对预测太过肯定的问题
        由于二分类(<5 and >=5)导致同类样本的特征值比较杂, 实际会很影响分类效果

    算法步骤:
    1. 加载训练集, 初始化w b, 初始化学习率α
    2. 训练: 根据6.1.3的对数似然函数进行梯度下降(line 65), 设定迭代次数
    3. 测试: 使用更新完的w b计算测试集样本为正例的概率(line 77), 如果>=0.5则预测为正例
    4. 统计测试结果, 分析指标

    """
    def __init__(self, file, learning_rate=1e-4, batch_size=200):
        data, labels = load_data(file)
        self.data, self.labels = np.array(preprocess_data(data)), np.array(preprocess_label(labels))
        self.n_sample, self.n_feature = np.shape(self.data)
        self.w = np.zeros(self.n_feature)    # 把b放进了w里, 因为preprocess_data, 所以不需要+1
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, iter, use_batch=False):
        """训练函数"""
        print("Start train!")
        for _ in tqdm(range(iter)):
            if use_batch:
                for i in range(0, self.n_sample, self.batch_size):
                    total_grad_w = np.zeros(self.n_feature)
                    for j in range(i, i+self.batch_size):
                        xi, yi = self.data[j], self.labels[j]
                        exp_wx = np.exp(np.dot(self.w, xi))
                        total_grad_w += -1 * yi * xi + (exp_wx * xi) / (1 + exp_wx)
                    mean_grad_w = total_grad_w / self.batch_size
                    self.w = self.w - self.learning_rate * mean_grad_w
            else:
                for i in range(self.n_sample):
                    xi, yi = self.data[i], self.labels[i]
                    # 梯度下降, 梯度是 -L(w)关于w的偏导, max L(w) = min -L(w)
                    exp_wx = np.exp(np.dot(self.w, xi))
                    grad_w = -1 * yi * xi + (exp_wx * xi) / (1 + exp_wx)
                    self.w = self.w - self.learning_rate * grad_w

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_pos, n_pred_pos, n_correct_pos, n_correct = 0, 0, 0, 0
        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            exp_wx = np.exp(np.dot(self.w, xi))
            prob_pred = exp_wx / (1 + exp_wx)
            if prob_pred >= 0.5:
                n_pred_pos += 1
            if yi == 1:
                n_real_pos += 1
            if yi == 1 and prob_pred >= 0.5:
                n_correct_pos += 1
            if (yi == 1 and prob_pred >= 0.5) or (yi == 0 and prob_pred < 0.5):
                n_correct += 1

        evaluate_binary_classify(n_correct, n_correct_pos, n_real_pos, n_pred_pos, n_sample)


def preprocess_data(data):
    """将b融合进w构成w', 为了维度对应也需要将xi增加一维, 即w' * x' = w * x + b * 1"""
    for i, xi in enumerate(data):
        # 这里一定要归一化！！255的指数幂会导致上溢出。这里debug花了好长时间
        for j, v in enumerate(xi):
            xi[j] = v / 255
        xi.append(1)
    return data


def preprocess_label(labels):
    """将Mnist数据集的label转为二分类任务"""
    for i, v in enumerate(labels):
        # logistic regression要求随机变量Y的取值为0或1, 不能取-1
        # 因为似然函数里面是p^yi * q^(1-yi)
        labels[i] = 0 if v < 5 else 1
    return labels


if __name__ == '__main__':
    start = time.time()

    my_logistic_regression = MyLogisticRegression('./Mnist/mnist_train.csv')
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.array(preprocess_data(test_data)), np.array(preprocess_label(test_labels)).T

    my_logistic_regression.train(iter=30, use_batch=False)
    my_logistic_regression.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
