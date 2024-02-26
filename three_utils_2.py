import pandas as pd
import numpy as np
from math import log2,log10
import json, re,os, sys
from keras.utils import to_categorical  # 独热编码
from sklearn.preprocessing import MinMaxScaler     # 标签 二值化
from keras import models, layers
import random
import tensorflow.keras as keras

from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve

# 根据已知基因对构建矩阵
def get_GRN(Ecoli_GRN_known,genename):
    '''
    输入基因名称列表和已知基因对名称，返回矩阵geneNetwork
    :param Ecoli_GRN_known: 已知TF-target gene对，TF,target gene, 1/0, types(activator, repressor, unknown)， 4329*3
    :param genename: 基因名称列表
    :return: geneNetwork
    '''

    rowNumber = []  # 存放 TF
    colNumber = []  # 存放 target gene
    regulation_types = []
    TF_name = np.array(Ecoli_GRN_known[0])  # TF 名称
    target_name = np.array(Ecoli_GRN_known[1])  # target gene 名称
    regulation_type = np.array(Ecoli_GRN_known[2])  # 调控类型

    genename2 = genename.tolist()

    for i in range(len(TF_name)):
        rowNumber.append(genename2.index(TF_name[i]))

    for i in range(len(target_name)):
        colNumber.append(genename2.index(target_name[i]))

    for i in range(len(regulation_type)):
        regulation_types.append(regulation_type[i])  # 2308个激活，2261个抑制，25个未知

    # 去除重复基因对
    # genepair = [] # 8907
    # for i in range(len(TF_name)):
    #     # temp = (rowNumber[i],colNumber[i],regulation_types[i])
    #     temp = (rowNumber[i], colNumber[i])
    #     genepair.append(temp)
    # new_genepair = list(set(genepair))
    # new_genepair.sort(key=genepair.index) # 8340


    num_activator = 0
    num_repressor = 0
    num_unknown = 0
    geneNetwork = np.zeros((len(genename2), len(genename2)))


    for i in range(len(regulation_types)):
        r = rowNumber[i]
        c = colNumber[i]
        if regulation_types[i] == 'activator':
            geneNetwork[r][c] = int(2.0)
            # num_activator += 1
        elif regulation_types[i] == 'repressor':
            geneNetwork[r][c] = int(1.0)
            # num_repressor += 1
        else:
            geneNetwork[r][c] = int(0.0)

    for i in range(geneNetwork.shape[0]):
        for j in range(geneNetwork.shape[0]):
            if geneNetwork[i][j] == 2:
                num_activator += 1
            elif geneNetwork[i][j] == 1:
                num_repressor += 1
            else:
                num_unknown += 1
    return geneNetwork, num_activator, num_repressor, num_unknown


def standard(rawdata):
    new_data1 = np.zeros((rawdata.shape[0],rawdata.shape[1]))
    for i in range(rawdata.shape[0]):
        for j in range(rawdata.shape[1]):
            new_data1[i][j] = log2(rawdata[i][j]+1)
    Standard_data = MinMaxScaler().fit_transform(new_data1)
    return Standard_data

def create_samples_histogram2d(EXP_cold, Ecoli_GRN,num_negative):

    EXP_cold_new = np.zeros((EXP_cold.shape[0], EXP_cold.shape[1]))
    for i in range(EXP_cold.shape[0]):
        for j in range(EXP_cold.shape[1]):
            EXP_cold_new[i][j] = log10(EXP_cold[i][j] + 10 ** -2)

    sample_cold_pos_2 = []
    sample_cold_pos_1 = []
    sample_cold_neg_0 = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            # # 转录因子的表达值
            # tf1 = EXP_cold_new[i]  # (24,)
            # # 靶基因的表达值
            # target1 = EXP_cold_new[j]  # (24,)
            #
            # H_T = np.histogram2d(tf1, target1, bins=32)
            # H = H_T[0].T
            # HT = np.zeros((H.shape[0], H.shape[1]))
            # for i in range(H.shape[0]):
            #     for j in range(H.shape[1]):
            #         HT[i][j] =(log10(H[i][j]/ len(tf1) + 10 ** -4)+4)/ 4

            # HT = (log10(H / len(tf1) + 10 ** -4) + 4) / 4

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)

                H_T = np.histogram2d(tf1, target1, bins=32)
                H = H_T[0].T
                HT = np.zeros((H.shape[0], H.shape[1]))
                for i in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        HT[i][j] = (log10(H[i][j] / len(tf1) + 10 ** -4) + 4) / 4

                sample_cold_pos_2.append(HT)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)

                H_T = np.histogram2d(tf1, target1, bins=32)
                H = H_T[0].T
                HT = np.zeros((H.shape[0], H.shape[1]))
                for i in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        HT[i][j] = (log10(H[i][j] / len(tf1) + 10 ** -4) + 4) / 4

                sample_cold_pos_1.append(HT)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(HT)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i,j)

    random.shuffle(negative_0_position)
    negative_0_position2 = negative_0_position[0:num_negative]
    for k in range(len(negative_0_position2)):
        A = negative_0_position2[k][0]
        B = negative_0_position2[k][1]
        # 转录因子的表达值
        tf1 = EXP_cold_new[A]  # (24,)
        # 靶基因的表达值
        target1 = EXP_cold_new[B]  # (24,)

        H_T = np.histogram2d(tf1, target1, bins=32)
        H = H_T[0].T
        HT = np.zeros((H.shape[0], H.shape[1]))
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                HT[i][j] = (log10(H[i][j] / len(tf1) + 10 ** -4) + 4) / 4

        labels_neg_0.append(0)
        sample_cold_neg_0.append(HT)

    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size = sample_cold_pos_2[0].shape
    print(feature_size)

    return positive2_data, positive1_data, negative0_data, feature_size

# 构建Ecoli cold样本（expression ）可以产生2205*2205个样本 包含时序基因表达值
def create_samples_concatenate(EXP_cold, Ecoli_GRN):
    '''
    先对表达值进行了标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''
    EXP_cold = standard(EXP_cold)  # 标准化
    sample_cold_pos_2 = []
    sample_cold_pos_1 = []
    sample_cold_neg_0 = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            # 转录因子的表达值
            tf1 = EXP_cold[i]  # (24,)
            # 靶基因的表达值
            target1 = EXP_cold[j]  # (24,)
            temp = np.hstack((tf1, target1))  # (48,)
    #         sample_cold.append(temp) # 241892个样本

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                sample_cold_pos_2.append(temp)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                sample_cold_pos_1.append(temp)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                sample_cold_neg_0.append(temp)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i,j)
    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size = sample_cold_pos_2[0].shape[0]
    print(feature_size)

    return positive2_data, positive1_data, negative0_data, feature_size


# 构建Ecoli cold样本（expression ）可以产生2205*2205个样本 包含时序基因表达值
def create_samples_CNNGRN_P(EXP_cold, Ecoli_GRN, num_negative):
    '''
    先对表达值进行了标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''
    EXP_cold = standard(EXP_cold)  # 标准化
    sample_cold_pos_2 = []
    sample_cold_pos_1 = []
    sample_cold_neg_0 = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            label = int(Ecoli_GRN[i][j])
            # print(label)
            if label == 2:
                # 转录因子的表达值
                tf1 = EXP_cold[i]  # (24,)
                tf_net = Ecoli_GRN[i]
                # 靶基因的表达值
                target1 = EXP_cold[j]  # (24,)
                target_net = Ecoli_GRN[j]
                temp1 = np.hstack((tf1, tf_net))  # (48,)
                temp2 = np.hstack((target1, target_net))  # (48,)
                temp = np.hstack((temp1, temp2))

                sample_cold_pos_2.append(temp)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # 转录因子的表达值
                tf1 = EXP_cold[i]  # (24,)
                tf_net = Ecoli_GRN[i]
                # 靶基因的表达值
                target1 = EXP_cold[j]  # (24,)
                target_net = Ecoli_GRN[j]
                temp1 = np.hstack((tf1, tf_net))  # (48,)
                temp2 = np.hstack((target1, target_net))  # (48,)
                temp = np.hstack((temp1, temp2))

                sample_cold_pos_1.append(temp)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(temp)
                # labels_neg_0.append(label)
                negative_0_positions.append((i, j))

    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]

        # 转录因子的表达值
        tf1 = EXP_cold[i]  # (24,)
        tf_net = Ecoli_GRN[i]
        # 靶基因的表达值
        target1 = EXP_cold[j]  # (24,)
        target_net = Ecoli_GRN[j]
        temp1 = np.hstack((tf1, tf_net))  # (48,)
        temp2 = np.hstack((target1, target_net))  # (48,)
        temp = np.hstack((temp1, temp2))
        sample_cold_neg_0.append(temp)
        labels_neg_0.append(0)

    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size = sample_cold_pos_2[0].shape[0]
    print(feature_size)

    return positive2_data, positive1_data, negative0_data, feature_size

def create_samples_dream5(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t):
    '''
    先对表达值进行标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''

    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(EXP_cold.shape[0]):
        for j in range(EXP_cold.shape[0]):
            # 转录因子的表达值
            tf1 = EXP_cold[i]  # (24,)
            # 转录因子的源邻居向量
            tf_s = GRN_embedding_s[i]  # (32,)
            # 转录因子的靶邻居向量
            tf_t = GRN_embedding_t[i]  # (32,)
            # 靶基因的表达值
            target1 = EXP_cold[j]  # (24,)
            # 靶基因的源邻居向量
            target_s = GRN_embedding_s[j]  # (32,)
            # 靶基因的靶邻居向量
            target_t = GRN_embedding_t[j]  # (32,)

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # sample_cold_pos_2.append(temp)
                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # sample_cold_pos_1.append(temp)
                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(temp)
                sample_cold_neg_0_tf.append(tf1)
                sample_cold_neg_0_target.append(target1)

                sample_cold_pos_0_net_tf_s.append(tf_s)
                sample_cold_pos_0_net_tf_t.append(tf_t)
                sample_cold_pos_0_net_target_s.append(target_s)
                sample_cold_pos_0_net_target_t.append(target_t)

                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets

def create_samples_concatenate_dream5(EXP_cold, Ecoli_GRN):
    '''
    先对表达值进行了标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''
    sample_cold_pos_2 = []
    sample_cold_pos_1 = []
    sample_cold_neg_0 = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            # 转录因子的表达值
            tf1 = EXP_cold[i]  # (24,)
            # 靶基因的表达值
            target1 = EXP_cold[j]  # (24,)
            temp = np.hstack((tf1, target1))  # (48,)
    #         sample_cold.append(temp) # 241892个样本

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                sample_cold_pos_2.append(temp)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                sample_cold_pos_1.append(temp)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                sample_cold_neg_0.append(temp)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i,j)
    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size = sample_cold_pos_2[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size

def create_samples_human_counts(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t):
    '''
    先对表达值进行标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''
    EXP_cold = standard(EXP_cold)  # 标准化
    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            # 转录因子的表达值
            tf1 = EXP_cold[i]  # (24,)
            # 转录因子的源邻居向量
            tf_s = GRN_embedding_s[i]  # (32,)
            # 转录因子的靶邻居向量
            tf_t = GRN_embedding_t[i]  # (32,)
            # 靶基因的表达值
            target1 = EXP_cold[j]  # (24,)
            # 靶基因的源邻居向量
            target_s = GRN_embedding_s[j]  # (32,)
            # 靶基因的靶邻居向量
            target_t = GRN_embedding_t[j]  # (32,)

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # sample_cold_pos_2.append(temp)
                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # sample_cold_pos_1.append(temp)
                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(temp)
                sample_cold_neg_0_tf.append(tf1)
                sample_cold_neg_0_target.append(target1)

                sample_cold_pos_0_net_tf_s.append(tf_s)
                sample_cold_pos_0_net_tf_t.append(tf_t)
                sample_cold_pos_0_net_target_s.append(target_s)
                sample_cold_pos_0_net_target_t.append(target_t)

                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets


# 获取特征和标签
def transform_data_cnn(train_data):
    """
    train_data的结构：[(sample,label,（坐标）)]
    return :
        trainX, labelY, position
    """
    feature = []  # 特征
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):
        feature.append(train_data[i][0])
        label_.append(train_data[i][1])
        position.append(train_data[i][2])

    feature = np.array(feature)
    dataX = feature[:,np.newaxis,:,np.newaxis]  # (6500, 1, 48, 1)

    print("the shape of feature: ",dataX.shape)

    label_ = np.array(label_)
    # print('label_:', label_)
    # 对标签 独热编码
    labelY = to_categorical(label_,3)
    # print('labelY: ',labelY)

    position = np.array(position)

    # text_save('label_.csv', label_)
    # text_save('labelY.csv', labelY)

    return dataX, labelY, position

# 获取特征和标签
def transform_data_resnet(train_data):
    """
    train_data的结构：[(sample,label,（坐标）)]
    return :
        trainX, labelY, position
    """
    feature = []  # 特征
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):
        feature.append(train_data[i][0])
        label_.append(train_data[i][1])
        position.append(train_data[i][2])

    feature = np.array(feature)
    # dataX = feature[:,np.newaxis,:,np.newaxis]  # (6500, 1, 48, 1)
    dataX = feature[:,np.newaxis,:]  # (6500, 1, 48)

    print("the shape of feature: ",dataX.shape)

    label_ = np.array(label_)
    # print('label_:', label_)
    # 对标签 独热编码
    labelY = to_categorical(label_,3)
    # print('labelY: ',labelY)

    position = np.array(position)

    # text_save('label_.csv', label_)
    # text_save('labelY.csv', labelY)

    return dataX, labelY, position

# 获取特征和标签
def transform_data_histogram2d(train_data):
    """
    train_data的结构：[(sample,label,（坐标）)]
    return :
        trainX, labelY, position
    """
    feature = []  # 特征
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):
        feature.append(train_data[i][0])
        label_.append(train_data[i][1])
        position.append(train_data[i][2])

    feature = np.array(feature)
    print("the shape of feature1: ", feature.shape)
    # dataX = feature[:,np.newaxis,:,np.newaxis]  # (6500, 1, 48, 1)
    dataX = feature[:, :,:,np.newaxis]  # (6500, 1, 48)

    print("the shape of feature2: ", dataX.shape)
    inputshape = dataX.shape

    label_ = np.array(label_)
    # print('label_:', label_)
    # 对标签 独热编码
    labelY = to_categorical(label_,3)
    # print('labelY: ',labelY)

    position = np.array(position)

    # text_save('label_.csv', label_)
    # text_save('labelY.csv', labelY)

    return dataX, labelY, position, inputshape


# 构建Ecoli CNN模型
def create_cnn_model(feature_size):
    # 搭建网络 ,无正则化
    model = models.Sequential()
    model.add(layers.Conv2D(16, (1, 7), border_mode='same', activation="relu",
                            input_shape=(1,feature_size, 1)))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling2D((1, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (1, 7), border_mode='same', activation="relu"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling2D((1, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64,(1,7),border_mode='same',activation="relu"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling2D((1, 2), padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation="softmax"))

    # 配置优化设置
    model.compile(optimizer='rmsprop',
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    return model




