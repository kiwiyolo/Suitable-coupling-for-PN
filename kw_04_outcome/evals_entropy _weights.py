# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/2/29 23:46
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : evals_entropy _weights
# @IDE     : PyCharm
# -----------------------------------------------------------------
import pandas as pd
import numpy as np


def entropy_method(matrix, directions, min_positive):
    num_samples, num_criteria = matrix.shape
    weights = np.zeros(num_criteria)
    e = g = np.zeros(num_criteria)
    for j in range(num_criteria):
        # 计算指标 j 的概率分布
        m_max = np.max(matrix[:, j])
        m_min = np.min(matrix[:, j])
        dv = m_max - m_min
        normal_w = None
        # 根据指标的方向进行标准化处理
        if directions[j] == 'positive':
            normal_w = (matrix[:, j] - m_min)/dv + min_positive

        elif directions[j] == 'negative':
            normal_w = (m_max - matrix[:, j])/dv + min_positive
        pij = normal_w / np.sum(normal_w)
        k = -1 / np.log(len(pij))
        e[j] = k * np.sum(np.multiply(pij, np.log(pij)))
        g[j] = 1 - e[j]


    # 归一化处理
    weights = g / np.sum(g)

    return weights

def main():
    # 评价指标矩阵示例，每一行代表一个样本，每一列代表一个评价指标
    # 从CSV文件读取数据
    leadtime_l = ['30', '60', '90']
    min_positive = 1e-6     # 用于计算权重时平移→解决0值问题
    file = f"./out_qu6_evals_熵值法_weights/evals_weights_out.csv"
    f = open(file, 'a')

    f.write("leadtime,POD,CSI,HSS,CC,ACC,FAR,MSE")
    f.write("\n")
    for lt in leadtime_l:
        data = pd.read_csv(f'.\out_qu6_evals_熵值法_weights\evals_weights_{lt}min.csv')

        # 提取评价指标矩阵和指标方向
        criteria_matrix = data.iloc[1:, :].values
        directions = data.iloc[0, :].values # 评价指标的方向，'positive'代表正向指标，'negative'代表反向指标

        # 使用熵值法计算权重
        weights = entropy_method(criteria_matrix.astype(float), directions, min_positive)

        print(f"{lt}min的评价指标权重：", weights)
        f.write(f"{lt}min,")
        for i in range(len(weights)):
            f.write(f"{weights[i]},")
        f.write("\n")
    f.close()
if __name__ == '__main__':
    main()