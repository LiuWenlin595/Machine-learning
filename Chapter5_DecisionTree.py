#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 15:18
# @Author  : Wenlin Liu
# @File    : Chapter5_DecisionTree.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果(ID3, 无剪枝, 特征值二分化):
准确率: 86.36%
精确率: 86.23%
召回率: 86.20%
运行时间: 193.33s

运行结果(C4.5, 无剪枝, 特征值二分化):
准确率: 86.17%
精确率: 86.06%
召回率: 86.02%
运行时间: 209.32s
"""
from utils import *
import numpy as np
import time
from tqdm import *
import collections


class MyDecisionTree:
    """实现决策树模型(ID3, C4.5)

    算法步骤:
    1. 加载数据集, 设定信息增益(比)的阈值, 设定特征选择区间和特征值选择区间
    2. 生成决策树: 在所有特征中选择信息增益(比)最大的特征, 根据特征值将训练集分成多个子集,
                依次递归生成树的叶结点, 直到满足算法5.2(1)(2)(4)   (line 106、87)
    3. 测试: 根据决策树为每个测试样本寻找合适的叶结点, 叶结点的类别即为该节点的类别(line 144)
    4. 统计测试结果, 分析指标

    """

    def __init__(self, file, tree_type="ID3"):
        data, labels = load_data(file)
        self.data, self.labels = np.array(preprocess(data)), np.array(labels)
        self.n_cls = 10
        self.gain_threshold = 0.1
        self.gain_ratio_threshold = 0.1
        self.n_feature = np.shape(self.data)[1]
        self.feature_range = 2  # 每个特征可选的值
        self.feature_list = [i for i in range(self.n_feature)]
        self.tree = self.create_tree(self.data, self.labels, self.feature_list[:], tree_type=tree_type)

    def cal_entropy(self, labels):
        """计算经验熵, 对应5.7"""
        ans = 0
        labels_set = set([i for i in labels])
        # 这样就不用考虑某一个类别样本数量为0的情况了
        for label in labels_set:
            p = labels[labels == label].size / labels.size
            ans += -1 * p * np.log2(p)
        return ans

    def cal_entropy_cond_feature(self, data, labels, feature):
        """计算经验条件熵, 对应5.8

        feature: 对应Mnist数据集的784维特征

        """
        ans = 0
        feature_array = data[:, feature]
        feature_set = set([i for i in feature_array])
        for i in feature_set:
            ratio = feature_array[feature_array == i].size / feature_array.size
            ans += ratio * self.cal_entropy(labels[feature_array == i])
        return ans

    def cal_gain_ratio(self, gain, data, feature):
        """计算信息增益比, 对应5.10"""
        feature_array = data[:, feature]
        feature_set = set([i for i in feature_array])
        entropy_with_feature = 0
        for i in feature_set:
            p = feature_array[feature_array == i].size / feature_array.size
            entropy_with_feature += -1 * p * np.log2(p)
        return gain / entropy_with_feature

    def choose_feature(self, data, labels, remain_feature, tree_type="ID3"):
        """为决策树选择最优的特征"""
        best_feature, max_target = -1, 0
        cur_entropy = self.cal_entropy(labels)
        if tree_type == "ID3":
            for feature in remain_feature:
                tmp_gain = cur_entropy - self.cal_entropy_cond_feature(data, labels, feature)
                if max_target < tmp_gain:
                    max_target = tmp_gain
                    best_feature = feature
        elif tree_type == "C4.5":
            for feature in remain_feature:
                tmp_gain = cur_entropy - self.cal_entropy_cond_feature(data, labels, feature)
                tmp_gain_ratio = self.cal_gain_ratio(tmp_gain, data, feature)
                if max_target < tmp_gain_ratio:
                    max_target = tmp_gain_ratio
                    best_feature = feature
        return best_feature, max_target

    def create_tree(self, data, labels, remain_feature, tree_type="ID3"):
        """构建决策树, 前序遍历实现特征选择和样本划分"""
        n_data = np.shape(data)[0]
        labels_set = set([i for i in labels])
        if len(labels_set) == 1:    # 算法5.2 (1)
            return labels_set.pop()
        # 一般来讲，训练集必须大到包含所有特征的所有可选值，否则测试集遇到决策树中特征未分类的值就没办法分类了
        # 所以这里不需要考虑n_data == 0的情况
        if len(data[0]) == 0:   # 算法5.2 (2)
            return classify(labels)

        cur_feature, max_target = self.choose_feature(data, labels, remain_feature, tree_type)
        print("choose feature: " + str(cur_feature) + ", n_sample: " + str(n_data))
        if tree_type == "ID3" and max_target < self.gain_threshold:    # 算法5.2 (4)
            return classify(labels)
        elif tree_type == "C4.5" and max_target < self.gain_ratio_threshold:
            return classify(labels)

        data_list, label_list = [[] for _ in range(self.feature_range)], [[] for _ in range(self.feature_range)]
        for i in range(n_data):
            data_list[data[i][cur_feature]].append(data[i])
            label_list[data[i][cur_feature]].append(labels[i])

        tree = {cur_feature: {}}    # 这里借鉴了一下github
        remain_feature.remove(cur_feature)
        for i in range(self.feature_range):
            tree[cur_feature][i] = self.create_tree(np.array(data_list[i]), np.array(label_list[i]), remain_feature[:])
        return tree

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_list, n_pred_list, n_correct_list = np.zeros(self.n_cls), np.zeros(self.n_cls), np.zeros(self.n_cls)

        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            tree = self.tree
            while isinstance(tree, dict):
                feature = list(tree.keys())[0]  # 这里要转成list, 因为dic_keys不能取索引
                sub_tree = tree[feature][xi[feature]]
                tree = sub_tree
            # 预测以及统计
            pred = tree
            n_real_list[yi] += 1
            n_pred_list[pred] += 1
            if yi == pred:
                n_correct_list[yi] += 1

        evaluate(n_correct_list, n_real_list, n_pred_list, n_sample)


def classify(labels):
    """将叶结点设为样本中得票数最多的类别"""
    cls_dict = collections.defaultdict(int)
    for label in labels:
        cls_dict[label] += 1
    return max(cls_dict, key=cls_dict.get)


def preprocess(data):
    """将Mnist数据集特征预处理, 以减少特征取值的数量"""
    n = np.shape(data)[0]
    for i in range(n):
        for j, v in enumerate(data[i]):
            data[i][j] = data[i][j] // 128
    return data


if __name__ == "__main__":
    start = time.time()

    my_decision_tree = MyDecisionTree('./Mnist/mnist_train.csv', tree_type="ID3")
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.array(preprocess(test_data)), np.array(test_labels)

    my_decision_tree.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
