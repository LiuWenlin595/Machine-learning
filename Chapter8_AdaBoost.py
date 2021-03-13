#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 20:19
# @Author  : Wenlin Liu
# @File    : Chapter8_AdaBoost.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果:(n_tree = 40, 特征值二值化, 标签二值化)
准确率: 97.98%
精确率: 98.92%
召回率: 98.84%
运行时间: 32444.18s

运行结果:(n_tree = 40, 特征值二值化, 标签 = 0 if label = 0 else 1)
准确率: 79.71%
精确率: 78.91%
召回率: 79.51%
运行时间: 32336.28s
"""
from utils import *
import numpy as np
import time
from tqdm import *


class MyAdaBoost:
    """实现AdaBoost分类模型

    算法步骤:
    1. 加载训练集, 特征值二分化, 二分类
    2. 生成AdaBoost Tree: 根据最小误差率的原则生成多个决策树桩 (line 69), 然后计算每个树的权重 (line 87),
        然后更新所有样本的权重, 错误样本权重增高, 正确样本权重降低 (line 89)
    3. 测试: 计算AdaBoost Tree的线性组合来对测试样本分类 (line 110)
    4. 统计测试结果, 分析指标

    """
    def __init__(self, file, n_tree=10):
        data, labels = load_data(file)
        self.data, self.labels = np.array(preprocess_data(data)), np.array(preprocess_label(labels))
        self.n_sample, self.n_feature = np.shape(self.data)

        self.n_tree = n_tree
        self.D = np.ones(self.n_sample) / self.n_sample
        self.feature_interval = [-0.5, 0.5, 1.5]
        self.rule = ["left_is_pos", "right_is_pos"]
        self.trees = self.generate_adaboost_trees()

    def cal_e(self, feature, interval, rule):
        """根据某一具体特征、特征分隔点 和 分隔规则 来计算分类误差率em, 对应公式8.1"""
        incorrect_rate, pred_list = 0, np.zeros(self.n_sample)
        rule_num = 1 if rule == "right_is_pos" else -1
        for i in range(self.n_sample):
            feature_value = self.data[i][feature]
            # 四种情况:
            # feature_value > interval and "right is pos" : +
            # feature_value < interval and "right is pos" : -
            # feature_value > interval and "left is pos"  : +
            # feature_value < interval and "left is pos"  : -
            pred = 1 if (feature_value - interval) * rule_num > 0 else -1
            pred_list[i] = pred
            if pred != self.labels[i]:
                incorrect_rate += self.D[i]
        return incorrect_rate, pred_list

    def generate_one_tree(self):
        """生成单个基本分类器, 实现算法8.1(2)(a)

        具体来讲就是实现一个决策树桩, 选择一个最优特征和最优分隔点可以最大化地把正负样本分隔开
        因为做了归一化, 所以每个特征值的选择都是0和1, 因此分隔点interval选为-0.5, 0.5, 1.5
        将正负样本区分开, 又可以细分为 小于分隔点为正例 或 大于分隔点为正例 的两个rule

        """
        tree = {'e': 1, 'pred_list': None, 'feature': None, 'interval': None, 'rule': None}    # 初始化
        for feature in range(self.n_feature):
            for interval in self.feature_interval:
                for rule in self.rule:
                    e, pred_list = self.cal_e(feature, interval, rule)
                    if e < tree['e']:
                        tree['e'] = e
                        tree['pred_list'] = pred_list   # 这里记录Gx, 公式8.4的时候就不用再算一遍了
                        tree['feature'] = feature
                        tree['interval'] = interval
                        tree['rule'] = rule
        return tree

    def generate_adaboost_trees(self):
        """生成n_tree个树, 即算法8.1(2), 对应公式8.2 8.3 8.4 8.5"""
        print("generate AdaBoost trees!")
        trees = []
        for _ in tqdm(range(self.n_tree)):
            tree = self.generate_one_tree()
            trees.append(tree)

            tree['alpha'] = np.log((1 - tree['e']) / tree['e']) / 2

            self.D = self.D * np.exp(-tree['alpha'] * self.labels * tree['pred_list'])
            self.D = self.D / sum(self.D)

        return trees

    def predict(self, xi):
        """根据AdaBoost模型预测结果, 对应公式8.6 8.7"""
        pred = 0
        for tree in self.trees:
            rule_num = 1 if tree['rule'] == "right_is_pos" else -1
            feature_value = xi[tree['feature']]
            sub_pred = 1 if (feature_value - tree['interval']) * rule_num > 0 else -1
            pred += tree['alpha'] * sub_pred
        return 1 if pred > 0 else -1

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_pos, n_pred_pos, n_correct_pos, n_correct = 0, 0, 0, 0
        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            pred = self.predict(xi)
            if pred > 0:
                n_pred_pos += 1
            if yi > 0:
                n_real_pos += 1
            if yi > 0 and pred > 0:
                n_correct_pos += 1
            if yi * pred > 0:
                n_correct += 1

        evaluate_binary_classify(n_correct, n_correct_pos, n_real_pos, n_pred_pos, n_sample)


def preprocess_data(data):
    """对训练样本做归一化操作, 因为涉及到exp操作"""
    for i, xi in enumerate(data):
        for j, v in enumerate(xi):
            xi[j] = 0 if v < 128 else 1
    return data


def preprocess_label(labels):
    """将Mnist数据集的label转为二分类任务"""
    for i, v in enumerate(labels):
        labels[i] = -1 if v == 0 else 1
    return labels


if __name__ == '__main__':
    start = time.time()

    my_adaboost = MyAdaBoost('./Mnist/mnist_train.csv', n_tree=40)
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.array(preprocess_data(test_data)), np.array(preprocess_label(test_labels)).T

    my_adaboost.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))