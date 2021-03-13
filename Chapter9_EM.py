#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 19:52
# @Author  : Wenlin Liu
# @File    : Chapter9_EM.py
"""
数据集: np.random.normal实现的多个高斯分布数据集的混合
测试集数量: 10000
-------------------------
运行结果:(两个高斯分布的混合, 迭代500次)
predict0: miu=-2.0, sigma=0.52, alpha=0.3
true0: miu=-2, sigma=0.5, alpha=0.3
predict1: miu=0.52, sigma=0.99, alpha=0.7
true1: miu=0.5, sigma=1, alpha=0.7
运行时间: 1.23s

运行结果:(五个高斯分布的混合, 迭代5000次, 没有调初值)
predict0: miu=-2.0, sigma=0.5, alpha=0.29
true0: miu=-2, sigma=0.5, alpha=0.3
predict1: miu=0.57, sigma=1.1, alpha=0.34
true1: miu=0.5, sigma=1, alpha=0.3
predict2: miu=3.2, sigma=1.4, alpha=0.11
true2: miu=3, sigma=1.5, alpha=0.2
predict3: miu=0.74, sigma=2.2, alpha=0.13
true3: miu=6, sigma=2.5, alpha=0.1
predict4: miu=5.4, sigma=2.6, alpha=0.13
true4: miu=0, sigma=2, alpha=0.1
运行时间: 16.97s
"""
from utils import *
import numpy as np
import time
from tqdm import *
import math


class MyEM:
    """实现EM算法

    算法步骤:
    1. 创建测试集, 设定参数, 将k个高斯分布数据集混合到一起(line 109)
    2. 初始化k个高斯模型的均值、方差、权重, 初值的设定很敏感, 不能保证找到全局最优解 (line 52)
    3. 计算E步 (line 72), 计算M步 (line 80), 迭代一定次数
    4. 打印更新后预测的参数, 和真实参数比较 (line 97)

    """
    def __init__(self, data, n_model):
        self.data = data
        self.n_sample = np.shape(self.data)[0]

        # 设初值, EM算法对初值很敏感, 需要多试几次找到合适的初值
        self.n_model = n_model
        self.miu = np.array([1, 2, 3, 4, 5])
        self.sigma = np.array([1, 0.5, 1.5, 2.5, 2])
        self.alpha = np.ones(self.n_model) / self.n_model

    def cal_gauss(self):
        """计算单个高斯分布密度公式, 对应公式9.25

        这里直接做二维的向量运算, 最终得到的res是个矩阵
        行表示n_sample, 列表示n_model

        """

        res = 1 / (np.sqrt(2 * np.pi) * self.sigma) * \
              np.exp(-1 * (self.data - self.miu)**2 / (2 * self.sigma**2))
        return res

    def e_step(self):
        """混合高斯模型EM算法的E步, 没必要求Q函数, 求出响应度γ就可以了, 对应公式9.29上面的式子"""
        # 这里用到的都是向量和矩阵的乘除运算, 需要细心
        gamma = self.alpha * self.cal_gauss()
        sum_model = np.sum(gamma, axis=1)[:, np.newaxis]   # 对k求累加
        gamma /= sum_model
        return gamma

    def m_step(self, gamma):
        """混合高斯模型EM算法的M步, 对应公式9.30 9.31 9.32"""
        # 同样的, 用到的都是向量和矩阵的乘除运算, 需要细心
        sum_sample = np.sum(gamma, axis=0)  # 对j求累加
        # 这里卡了一会, sigma的更新会用到旧的miu, 所以必须要先更新sigma才能更新miu, 还一直以为自己numpy写错了
        self.sigma = np.sqrt(np.sum(np.multiply(gamma, np.square(self.data - self.miu)), axis=0) / sum_sample)
        # 这里会把miu的shape从(n_model,)变为(1, n_model), 执行test的时候需要注意一下
        self.miu = np.dot(self.data.T, gamma) / sum_sample
        self.alpha = sum_sample / self.n_sample

    def train(self, iter=500):
        """训练函数, 使用迭代次数代替收敛"""
        print("Start train!")
        for _ in tqdm(range(iter)):
            self.m_step(self.e_step())

    def test(self, miu, sigma, alpha):
        """测试函数"""
        print("Start test!")
        for i in range(self.n_model):
            print("predict{}: miu={:.2}, sigma={:.2}, alpha={:.2}".format
                  (i, self.miu[0, i], self.sigma[i], self.alpha[i]))
            print("true{}: miu={}, sigma={}, alpha={}".format(i, miu[i], sigma[i], alpha[i]))
            print()


def generate_data(miu, sigma, alpha, size, n_model):
    """生成n个高斯分布数据集, 混合成一个混合高斯模型"""
    data = []
    for i in range(n_model):
        tmp = np.random.normal(miu[i], sigma[i], int(size*alpha[i]))
        data.append(tmp)
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    data = data[:, np.newaxis]
    return data


if __name__ == '__main__':
    start = time.time()

    miu = [-2, 0.5, 3, 6, 0]
    sigma = [0.5, 1, 1.5, 2.5, 2]
    alpha = [0.3, 0.3, 0.2, 0.1, 0.1]
    data = generate_data(miu, sigma, alpha, size=10000, n_model=len(alpha))

    my_em = MyEM(data, n_model=len(alpha))
    my_em.train(5000)

    my_em.test(miu, sigma, alpha)

    end = time.time()
    print("运行时间: %.2fs" % (end - start))