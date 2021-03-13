#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 22:31
# @Author  : Wenlin Liu
# @File    : Chapter7_SVM.py
"""
数据集: Mnist
训练集数量: 60000
测试集数量: 10000
-------------------------
运行结果:(取200个训练样本, 标签=1 if label=0 else -1)
准确率: 95.76%
精确率: 72.42%
召回率: 91.63%
运行时间: 116.30s
"""
from utils import *
import numpy as np
import time
from tqdm import *


class MySVM:
    """实现非线性SVM模型

    算法步骤:
    1. 加载训练集, 初始化拉格朗日乘子α 偏置b 松弛变量系数C 高斯核标准差σ
    2. 预处理, 计算所有xi和xj的核函数
    3. 训练: 设定迭代次数, 采用对偶算法转换最优化问题, 采用SMO算法求解最优化问题 (line 137),
        更新α和b (line 162), 更新完后wx+b(g_xi)就可以直接计算出来 (line 66)
    4. 测试: 使用更新完的α b计算g_xi, g_xi的正负即是测试集样本的类别 (line 177)
    5. 统计测试结果, 分析指标

    """
    def __init__(self, file, c, sigma):
        data, labels = load_data(file)
        self.data, self.labels = np.array(preprocess_data(data[:200])), np.array(preprocess_label(labels[:200]))
        self.n_sample, self.n_feature = np.shape(self.data)

        self.C = c                              # 松弛变量的系数C
        self.sigma = sigma                      # 高斯核标准差σ
        self.alpha = np.zeros(self.n_sample)    # 拉格朗日乘子α
        self.b = 0                              # 偏置b

        self.kernel = self.cal_all_kernel()
        # 小优化, 刚开始α都是0, 所以Ei = -yi
        self.E = -1 * self.labels

    def cal_gauss_kernel(self, x, z):
        """采用高斯核计算K(x, z), 对应公式7.90"""
        return np.exp(-1 * np.sum(np.square(x - z)) / (2 * self.sigma ** 2))

    def cal_all_kernel(self):
        """计算所有的K(xi, xj)"""
        print("calculate kernel function!")
        kernel = np.zeros((self.n_sample, self.n_sample))
        for i in tqdm(range(self.n_sample)):
            for j in range(i, self.n_sample):
                value = self.cal_gauss_kernel(self.data[i], self.data[j])
                kernel[i][j] = value
                kernel[j][i] = value
        return kernel

    def cal_g_xi(self, xi):
        """计算SMO算法中定义的g(xi), 实际上就是wx+b, 对应公式7.104"""
        g_xi = self.b
        for i in range(self.n_sample):
            g_xi += self.alpha[i] * self.labels[i] * self.cal_gauss_kernel(self.data[i], xi)
        return g_xi

    def update_e(self):
        """用于SMO算法第二个变量的选择, 对应公式7.105, 为了节省计算将所有Ei值保存在列表中, 在每次更新α b后执行Ei的更新"""
        e = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            e[i] = self.cal_g_xi(self.data[i]) - self.labels[i]
        return e

    def is_satisfy_kkt(self, i):
        """判断第i个样本是否满足KKT条件, 在SMO算法中寻找需要更新的第一个变量时使用, 对应公式7.111 7.112 7.113"""
        # 小优化, 直接用g(xi) = Ei + yi就可以不用算g(xi)了
        g_xi = self.E[i] + self.labels[i]
        if self.alpha[i] == 0 and self.labels[i] * g_xi >= 1:
            return True
        if 0 < self.alpha[i] < self.C and self.labels[i] * g_xi == 1:
            return True
        if self.alpha[i] == self.C and self.labels[i] * g_xi <= 1:
            return True
        return False

    def get_alpha1(self):
        """得到SMO算法中需要更新的第一个变量

        选取所有样本中违反KKT条件最严重的样本, 实际代码中如果为了寻找违反KKT条件最严重的样本就需要遍历所有样本
        为了减少计算时间这里选择第一个违反KKT条件的样本

        """
        index1 = -1
        for i in range(self.n_sample):
            if not self.is_satisfy_kkt(i):
                index1 = i
                break
        if index1 == -1:
            return "ok", -1
        return self.alpha[index1], index1

    def get_alpha2(self, index1):
        """寻找SMO算法中需要更新的第二个变量

        原则是|E1-E2|越大越好, 课本里的方法是E1为正就选择minE2, E1为负就选择maxE2
        这里就采用这种思路, 对其中一些未包含的极小概率发生的事件做了异常处理

        """
        e1 = self.E[index1]
        if e1 > 0:
            index2 = np.argmin(self.E)
        elif e1 < 0:
            index2 = np.argmax(self.E)
        else:  # e1 == 0
            print("error: e1 = 0")
            index2 = np.argmax(self.E)
            # raise Exception("error: e1 = 0")
        if index1 == index2:
            print("error: index1 = index2")
            # raise Exception("error: index1 = index2")
        return self.alpha[index2], index2

    def train(self, iter=30):
        """训练函数

        顺着课本P144 P145 P148一行一行把代码捋下来就好了
        更新alpha的部分对应公式7.106 7.107 7.108 7.109
        更新b的部分对应公式7.115 7.116 7.117

        """
        print("Start train!")
        for _ in tqdm(range(iter)):
            a1_old, index1 = self.get_alpha1()
            if a1_old == "ok":
                print("iter: " + str(iter) + ", 所有变量满足KKT条件, 提前终止迭代！")
                break
            a2_old, index2 = self.get_alpha2(index1)
            y1, y2 = self.labels[index1], self.labels[index2]
            e1, e2 = self.E[index1], self.E[index2]
            k11, k22, k12 = self.kernel[index1][index1], self.kernel[index2][index2], self.kernel[index1][index2]

            if y1 != y2:
                low = max(0, a2_old - a1_old)
                high = min(self.C, self.C + a2_old - a1_old)
            else:  # y1 == y2
                low = max(0, a2_old + a1_old - self.C)
                high = min(self.C, a2_old + a1_old)

            a2_new_unc = a2_old + y2 * (e1 - e2) / (k11 + k22 - 2 * k12)
            if a2_new_unc > high:
                a2_new = high
            elif a2_new_unc < low:
                a2_new = low
            else:  # L < a2_new_unc < R
                a2_new = a2_new_unc
            a1_new = a1_old + self.labels[index1] * self.labels[index2] * (a2_old - a2_new)

            self.alpha[index1] = a1_new
            self.alpha[index2] = a2_new

            b1_new = -e1 - y1 * k11 * (a1_new - a1_old) - y2 * k12 * (a2_new - a2_old) + self.b
            b2_new = -e2 - y1 * k12 * (a1_new - a1_old) - y2 * k22 * (a2_new - a2_old) + self.b
            self.b = (b1_new + b2_new) / 2
            self.E = self.update_e()

    def test(self, data, labels):
        """测试函数"""
        print("Start test!")
        n_sample = np.shape(data)[0]
        n_real_pos, n_pred_pos, n_correct_pos, n_correct = 0, 0, 0, 0
        for i in tqdm(range(n_sample)):
            xi, yi = data[i], labels[i]
            pred = self.cal_g_xi(xi)
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
            xi[j] = v / 255
    return data


def preprocess_label(labels):
    """将Mnist数据集的label转为二分类任务"""
    for i, v in enumerate(labels):
        labels[i] = 1 if v == 0 else -1
    return labels


if __name__ == '__main__':
    start = time.time()

    my_svm = MySVM('./Mnist/mnist_train.csv', c=200, sigma=10)
    test_data, test_labels = load_data('./Mnist/mnist_test.csv')
    test_data, test_labels = np.array(preprocess_data(test_data)), np.array(preprocess_label(test_labels)).T

    my_svm.train(iter=20)

    middle = time.time()
    print("训练时间: %.2fs" % (middle - start))

    my_svm.test(test_data, test_labels)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))