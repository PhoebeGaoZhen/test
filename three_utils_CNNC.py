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

def get_Histogram(tf1, target1):
    H_T = np.histogram2d(tf1, target1, bins=32)
    H = H_T[0].T
    HT = np.zeros((H.shape[0], H.shape[1]))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            HT[i][j] = (log10(H[i][j] / len(tf1) + 10 ** -4) + 4) / 4

    return HT

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

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)
                HT = get_Histogram(tf1, target1)

                sample_cold_pos_2.append(HT)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)
                HT = get_Histogram(tf1, target1)

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

        HT = get_Histogram(tf1, target1)

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

def get_top_cov_pairs(EXP_cold, geneA, geneB, cov_or_corr="cov"):
    cov_matrix = np.cov(EXP_cold)

    # get cov value first
    if self.corr_matrix is None or self.cov_matrix is None:
        self.calculate_cov()
    if cov_or_corr == "corr":
        np.fill_diagonal(self.corr_matrix, 0)

    histogram_list = []
    networki = geneA.split(":")[0]

    x = get_histogram_bins(geneA, geneA)

    if add_self_image:
        histogram_list.append(x)

    x = get_histogram_bins(geneB, geneB)

    if add_self_image:
        histogram_list.append(x)

    index = get_index_by_networki_geneName(geneA)
    if cov_or_corr == "cov":
        cov_list_geneA = cov_matrix[index, :]
    else:
        cov_list_geneA = corr_matrix[index, :]
    cov_list_geneA = cov_list_geneA.ravel()
    if get_abs:
        cov_list_geneA = np.abs(cov_list_geneA)
    the_order = np.argsort(-cov_list_geneA)
    select_index = the_order[0:top_num]
    for j in select_index:
        # if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
        x = get_histogram_bins(geneA, str(j))

        histogram_list.append(x)
    ####
    indexB = get_index_by_networki_geneName(geneB)
    if cov_or_corr == "cov":
        cov_list_geneB = cov_matrix[indexB, :]
    else:
        cov_list_geneB = corr_matrix[indexB, :]
    cov_list_geneB = cov_list_geneB.ravel()
    if get_abs:
        cov_list_geneB = np.abs(cov_list_geneB)
    the_order = np.argsort(-cov_list_geneB)
    select_index = the_order[0:top_num]
    for j in select_index:
        # if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
        x = get_histogram_bins(str(j), geneB)

        histogram_list.append(x)
    return histogram_list

def create_samples_DeepDRIM(EXP_cold, Ecoli_GRN,num_negative):

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

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)
                HT = get_Histogram(tf1, target1)
                HT_N = get_top_cov_pairs(tf1, target1)

                sample_cold_pos_2.append(HT)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # 转录因子的表达值
                tf1 = EXP_cold_new[i]  # (24,)
                # 靶基因的表达值
                target1 = EXP_cold_new[j]  # (24,)
                HT = get_Histogram(tf1, target1)

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

        HT = get_Histogram(tf1, target1)

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

