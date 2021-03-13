#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 16:25
# @Author  : Wenlin Liu
# @File    : utils.py

import numpy as np
from tqdm import *


def load_data(file):
    """加载数据集"""
    print("loading data!")
    data, labels = [], []
    f = open(file, 'r')
    for line in tqdm(f.readlines()):
        cur_line = line.strip().split(',')
        # Mnist数据集都是str, 需要转成int
        labels.append(int(cur_line[0]))
        # 没有做归一化, 需要归一化处理的操作在各个算法里独自实现
        data.append([int(i) for i in cur_line[1:]])
    return data, labels


def cal_dist_2_norm(data1, data2):
    """计算两个样本的2范数距离"""
    return np.sqrt(np.sum(np.square(data1 - data2)))


def evaluate(n_correct_list, n_real_list, n_pred_list, n_sample):
    """评估函数, 评价模型的效果

    评估指标:
    准确率: 估计正确样本个数 / 总样本数
    精确率: mean(某一类别估计正确的样本个数 / 估计结果为该类的总样本数)
    召回率: mean(某一类别估计正确的样本个数 / 真实结果为该类的总样本数)

    """
    accuracy = np.sum(n_correct_list) / n_sample
    precision = np.mean(n_correct_list / n_pred_list)
    recall = np.mean(n_correct_list / n_real_list)
    print("准确率: %.2f" % (accuracy*100) + "%")
    print("精确率: %.2f" % (precision*100) + "%")
    print("召回率: %.2f" % (recall*100) + "%")


def evaluate_binary_classify(n_correct, n_correct_pos, n_real_pos, n_pred_pos, n_sample):
    """评估函数, 评价模型的效果

    针对二分类任务, 如感知机、逻辑斯谛回归、SVM等
    二分类任务的评估指标通常只关注正类

    """
    accuracy = n_correct / n_sample
    precision = n_correct_pos / n_pred_pos
    recall = n_correct_pos / n_real_pos
    print("准确率: %.2f" % (accuracy*100) + "%")
    print("精确率: %.2f" % (precision*100) + "%")
    print("召回率: %.2f" % (recall*100) + "%")

