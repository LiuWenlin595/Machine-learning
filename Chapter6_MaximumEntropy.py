#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 14:52
# @Author  : Wenlin Liu
# @File    : Chapter6_MaximumEntropy.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果(特征值二值化, 标签二值化, 训练样本20000):
准确率: 79.30%
精确率: 79.28%
召回率: 79.30%
运行时间: 30736.97s

运行结果(特征值二值化, 标签二值化, 训练样本60000)
准确率: 79.48%
精确率: 79.47%
召回率: 79.49%
运行时间: 92735.71s

运行结果(特征值二值化, 标签=0 if label=0 else 1, 训练样本20000)
M = 5000
准确率: 98.42%
精确率: 97.50%
召回率: 93.39%
运行时间: 28617.43s

M = 10000
准确率: 97.40%
精确率: 97.68%
召回率: 87.42%
运行时间: 30395.34s

M = 20000
准确率: 95.14%
精确率: 97.25%
召回率: 75.30%
运行时间: 28910.70s
"""

from utils import *
import numpy as np
import time
from tqdm import *
import collections


class MyMaximumEntropy:
    """实现最大熵模型, 用于求解含有不确定信息时样本的概率估计

    算法步骤:
    1. 加载数据集, 有必要的话做预处理, 设定类别数, 设定尺度迭代法的f# (实际上f#的值不确定, 是需要计算的, 但是为了简便固定使用10000,
        同时也测试了取5000和20000的情况, 准确率和精确率都差不多, M增大召回率降低明显, 可能跟二分类样本不均衡有关)
    2. 做一些前期准备, 记录所有特征函数 (line 83), 为了简化计算和避免混淆将特征函数映射到id (line 104),
        记录每个特征函数的关于训练集合P(X, Y)的期望值 (line 143), 记录每个特征函数关于训练集合P(X)和模型P(Y|X)的期望值 (line 159)
    3. 训练: 有了显式的P(y|x)计算公式后, 就可以求解min -H(P) = Σ P~(x) * P(y|x) * np.log(P(y|x))的最优化问题了
        实际操作则是直接根据尺度迭代法计算每个参数的更新量, 迭代iter次 (line 178)
    4. 测试: 取迭代后的参数, 根据所有的特征函数计算测试样本属于每个类别的概率, 取最大概率作为预测类别 (line 191)
    5. 统计测试结果, 分析指标
    """
    def __init__(self, file):
        data, labels = load_data(file)
        self.data, self.labels = preprocess_data(data[:20000]), preprocess_label(labels[:20000])
        self.n_cls = 2
        self.n_sample, self.n_feature = np.shape(self.data)
        self.feature_func_list, self.n_func = self.get_feature_func_list()
        self.func2id_list, self.id2func_dict = self.get_mapping_dict()
        self.exp_func_about_experience = self.cal_exp_feature_func_about_experience()
        self.w = np.zeros(self.n_func)
        self.M = 10000

    def get_feature_func_list(self):
        """统计所有的特征函数, 并用list(dict)进行收集

        list的维度表示每一维特征, dict的key表示(feature_value, label), value表示该特征对出现的次数 (value在计算E_p~(f)的时候用到)
        注意:每一个(feature_value, label)都对应一个特征函数, 所以特征函数的数量应该是n_feature * feature_range * n_cls
        但是有很多(feature_value, label)出现次数为0
        这里只考虑出现过的(feature_value, label), 因为后续我们会用一个映射字典对特征函数的数量进行缩减

        """
        feature_func_list = [collections.defaultdict(int) for _ in range(self.n_feature)]
        for i in range(self.n_sample):
            for feature in range(self.n_feature):
                # 将(xi[feature], yi)作为键存进feature_func_list[feature]的字典
                feature_func_list[feature][(self.data[i][feature], self.labels[i])] += 1
        n_func = 0   # 统计所有的(feature_value, label)的个数
        for dic in feature_func_list:
            n_func += len(dic)
        return feature_func_list, n_func

    def get_mapping_dict(self):
        """构建两个映射字典

        因为在训练集中并不是所有的(feature_value, label)都存在, 所以有很多(feature_value, label)的统计值为0
        另一方面, 因为不同的feature可能有相同的(value, label), 所以不能单纯的用字典记录(value, label)
        这里引入id来记录所有值不为0的(feature_value, label), 一方面可以减少值为0导致的不必要的计算, 另一方面可以防止(value, label)的混乱
        func2id_list:表示(feature_value, label) -> index
        id2func_dict:表示index -> [feature, (feature_value, label)]

        """
        # func2id_dict中需要用list记录784的维度, 因为不同的feature可能有相同的(value, label)
        func2id_list = [{} for _ in range(self.n_feature)]
        id2func_dict, index = {}, 0
        for feature in range(self.n_feature):
            for k in self.feature_func_list[feature].keys():
                # 这里的k其实就是(feature_value, label)
                func2id_list[feature][k] = index
                id2func_dict[index] = [feature, k]
                index += 1
        return func2id_list, id2func_dict

    def cal_prob_y_cond_x(self, x):
        """计算最大熵模型的终极目标P(y|x), 所有y的概率以列表形式存储, 对应书中公式6.22

        这里计算的时候做了归一化, 所以满足了概率和为1的约束, 拉格朗日乘子w0自动失效, 所以更新公式里没涉及到参数w0

        """
        prob_y_cond_x, z = np.zeros(self.n_cls), 0   # 先通过for循环计算分子, 然后再除以分母Z
        for i in range(self.n_func):
            feature, (feature_value, label) = self.id2func_dict[i]
            if x[feature] == feature_value:
                prob_y_cond_x[label] += self.w[i]
        # # 两种遍历方式都可以, 个人感觉这种复杂度要小一些, 但是上面的更容易理解一些
        # for feature in range(self.n_feature):
        #     for cls in range(self.n_cls):
        #         if (x[feature], cls) in self.func2id_list[feature]:
        #             index = self.func2id_list[feature][(x[feature], cls)]
        #             prob_y_cond_x[cls] += self.w[index]
        prob_y_cond_x = np.exp(prob_y_cond_x)
        z = np.sum(prob_y_cond_x)
        prob_y_cond_x = prob_y_cond_x / z
        return prob_y_cond_x

    def cal_exp_feature_func_about_experience(self):
        """计算特征函数f(x, y)关于经验分布P(X, Y)的期望值

        从特征函数的角度出发, 每次直接计算特征函数的期望
        这里需要把每一个特征函数的期望值都算出来, 迭代尺度算法的更新公式会用到

        """
        exp_func_about_experience = np.zeros(self.n_func)
        for i in range(self.n_func):
            feature, k = self.id2func_dict[i]
            # 每一个特征函数的x都是样本的某一维特征, 而特征x出现的总数就是训练样本的个数, 所以分母都是n_sample
            exp_func_about_experience[i] = self.feature_func_list[feature][k] / self.n_sample
        return exp_func_about_experience

    def cal_exp_feature_func_about_model(self):
        """计算特征函数f(x, y)关于模型P(Y|X)与经验分布P(X)的期望值

        从训练样本的角度出发, 每次从特征函数中寻找和样本特征匹配的那些, 只对这些特征函数的期望进行累加
        因为需要用到P(y|x), 而P(y|x)的计算需要样本x的全部特征, 不得已必须获得样本x
        所以就想了个笨法子, 遍历所有样本, 所以每次P~(feature_value)只+1, 表示当前的样本对当前的特征函数有贡献
        这里也需要把每一个特征函数的期望值也算出来, 迭代尺度算法的更新公式会用到

        """
        exp_func_about_model = np.zeros(self.n_func)
        for i in range(self.n_sample):
            xi = self.data[i]
            prob_y_cond_x_list = self.cal_prob_y_cond_x(xi)
            for feature in range(self.n_feature):
                for cls in range(self.n_cls):
                    if (xi[feature], cls) in self.func2id_list[feature].keys():
                        index = self.func2id_list[feature][(xi[feature], cls)]
                        exp_func_about_model[index] += (1 / self.n_sample) * prob_y_cond_x_list[cls]
            # # 这一种也是一样的实现, 上面的好理解一些, 而且复杂度应该更优一点
            # for index in range(self.n_cls):
            #     feature, (feature_value, label) = self.id2func_dict[index]
            #     if xi[feature] == feature_value:
            #         exp_func_about_model[index] += (1 / self.n_sample) * prob_y_cond_x_list[label]
        return exp_func_about_model

    def train(self, iter):
        """训练函数"""
        print("Start train!")
        for _ in tqdm(range(iter)):
            exp_func_about_model = self.cal_exp_feature_func_about_model()
            # 算法6.1(2)(a)
            delta_list = (1 / self.M) * np.log(self.exp_func_about_experience / exp_func_about_model)
            # 算法6.1(2)(b)
            self.w = self.w + delta_list

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_list, n_pred_list, n_correct_list = np.zeros(self.n_cls), np.zeros(self.n_cls), np.zeros(self.n_cls)
        for sample in tqdm(range(n_sample)):
            xi, yi = data[sample], labels[sample]
            pred_list = self.cal_prob_y_cond_x(xi)
            # 预测以及统计
            pred = np.argmax(pred_list)
            n_real_list[yi] += 1
            n_pred_list[pred] += 1
            if yi == pred:
                n_correct_list[yi] += 1

        evaluate(n_correct_list, n_real_list, n_pred_list, n_sample)


def preprocess_data(data):
    """将b融合进w构成w', 为了维度对应也需要将xi增加一维, 即w' * x' = w * x + b * 1"""
    for i, xi in enumerate(data):
        for j, v in enumerate(xi):
            xi[j] = 0 if v < 128 else 1
    return data


def preprocess_label(labels):
    """将Mnist数据集的label转为二分类任务"""
    for i, v in enumerate(labels):
        labels[i] = 0 if v == 0 else 1
    return labels


if __name__ == "__main__":
    start = time.time()

    my_maximum_entropy = MyMaximumEntropy('./Mnist/mnist_train.csv')
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = preprocess_data(test_data), preprocess_label(test_labels)

    my_maximum_entropy.train(iter=600)
    my_maximum_entropy.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))
