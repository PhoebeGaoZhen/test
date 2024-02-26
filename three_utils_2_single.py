import numpy as np
from math import log2
import re
from keras.utils import to_categorical  # 独热编码
from sklearn.preprocessing import MinMaxScaler     # 标签 二值化
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
import random
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
        regulation_types.append(regulation_type[i])

    # print(regulation_types)
    # print(len(rowNumber))
    # print(len(colNumber))
    # print(len(regulation_types))
    #
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

def createGRN_gene100(Ecoli_GRN_known,genename):
    rowNumber = []
    colNumber = []
    for i in range(len(Ecoli_GRN_known)):
        row = Ecoli_GRN_known[i][0]
        rownum = re.findall("\d+",row)
        rownumber = int(np.array(rownum))
        rowNumber.append(rownumber)

        col = Ecoli_GRN_known[i][1]
        colnum = re.findall("\d+",col)
        colnumber = int(np.array(colnum))
        colNumber.append(colnumber)

    geneNetwork = np.zeros((genename.shape[0],genename.shape[0]))
    for i in range(len(rowNumber)):
        r = rowNumber[i]-1
        c = colNumber[i]-1
        geneNetwork[r][c] = 1
#     print(np.sum(geneNetwork))
#     保存geneNetwork
#     data1 = pd.DataFrame(geneNetwork)
#     data1.to_csv('D:\jupyter_project\CNNGRN\DATA\DREAM100_samples\geneNetwork_100_'+str(net+1)+'.csv')
    return geneNetwork
def standard(rawdata):
    new_data1 = np.zeros((rawdata.shape[0],rawdata.shape[1]))
    for i in range(rawdata.shape[0]):
        for j in range(rawdata.shape[1]):
            new_data1[i][j] = log2(rawdata[i][j]+1)
    Standard_data = MinMaxScaler().fit_transform(new_data1)
    return Standard_data


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

    for i in range(2205):
        for j in range(2205):
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


def create_samples_single(EXP_cold, Ecoli_GRN):
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
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(2205):
        for j in range(2205):
            # 转录因子的表达值
            tf1 = EXP_cold[i]  # (24,)
            # 靶基因的表达值
            target1 = EXP_cold[j]  # (24,)
            # temp = np.hstack((tf1, target1))  # (48,)
            #         sample_cold.append(temp) # 241892个样本

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
                # sample_cold_pos_2.append(temp)
                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # sample_cold_pos_1.append(temp)
                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(temp)
                sample_cold_neg_0_tf.append(tf1)
                sample_cold_neg_0_target.append(target1)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    # 将 feature (sample) 与 label 绑在一起
    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, labels_neg_0, negative_0_position))  # len

    # print(len(positive2_data))
    # print(len(positive1_data))
    # print(len(negative0_data))
    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target

def create_samples_single_net(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):
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
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
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

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]
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

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)

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

def create_samples_dream5(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):
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
    negative_0_positions = []

    for i in range(EXP_cold.shape[0]):
        for j in range(EXP_cold.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
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

                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]
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

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)

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

def create_samples_human_counts(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):
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
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):
            # # 转录因子的表达值
            # tf1 = EXP_cold[i]  # (24,)
            # # 转录因子的源邻居向量
            # tf_s = GRN_embedding_s[i]  # (32,)
            # # 转录因子的靶邻居向量
            # tf_t = GRN_embedding_t[i]  # (32,)
            # # 靶基因的表达值
            # target1 = EXP_cold[j]  # (24,)
            # # 靶基因的源邻居向量
            # target_s = GRN_embedding_s[j]  # (32,)
            # # 靶基因的靶邻居向量
            # target_t = GRN_embedding_t[j]  # (32,)

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:
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

                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
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

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0_tf.append(tf1)
                # sample_cold_neg_0_target.append(target1)
                #
                # sample_cold_pos_0_net_tf_s.append(tf_s)
                # sample_cold_pos_0_net_tf_t.append(tf_t)
                # sample_cold_pos_0_net_target_s.append(target_s)
                # sample_cold_pos_0_net_target_t.append(target_t)

                labels_neg_0.append(label)
                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]

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

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)
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

def create_samples_human_FPKM(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t):
    '''
    先对表达值进行标准化
    :param EXP_cold:  时序基因表达值 2205*24
    :param Ecoli_GRN: GRN矩阵，2205*2205, 元素值包括3 2 1 0
    :return: 激活、阻遏、激活+阻遏、候选样本的列表，每个样本都包括特征和标签，特征是时序基因表达值，标签是3 2 1 0
    '''
    EXP_cold = MinMaxScaler().fit_transform(EXP_cold)
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
def transform_data_single(train_data):
    """
    train_data的结构：[(sampletf, sampletarget,label,（坐标）)]
    return :
        trainX, labelY, position
    """
    featuretf = []  # 特征
    featuretarget = []
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):
        featuretf.append(train_data[i][0])
        featuretarget.append(train_data[i][1])
        label_.append(train_data[i][2])
        position.append(train_data[i][3])

    featuretf = np.array(featuretf)
    featuretarget = np.array(featuretarget)
    # dataX = feature[:,np.newaxis,:,np.newaxis]  # (6500, 1, 48, 1)
    dataX_tf = featuretf[:,np.newaxis,:]  # (6500, 1, 48) (6500, 1, 24)
    dataX_target = featuretarget[:,np.newaxis,:]  # (6500, 1, 48) (6500, 1, 24)
    print("the shape of dataX_tf: ",dataX_tf.shape)
    print("the shape of dataX_target: ",dataX_target.shape)


    label_ = np.array(label_)
    # print('label_:', label_)
    # 对标签 独热编码
    labelY = to_categorical(label_,3)
    # print('labelY: ',labelY)

    position = np.array(position)

    # text_save('label_.csv', label_)
    # text_save('labelY.csv', labelY)

    return dataX_tf, dataX_target, labelY, position


def transform_data_single_net(train_data):
    """
    train_data的结构：[(expressiontf, expressiontarget, net_tf_s, net_tf_t, net_target_s, net_target_t,label,（坐标）)]
    return :
        expressiontf(dataX_tf), expressiontarget(dataX_target), net_tf_s, net_tf_t, net_target_s, net_target_t,label,position
    """
    featuretf_exp = []  # 特征
    featuretarget_exp = []
    net_tf_s = []
    net_tf_t = []
    net_target_s = []
    net_target_t = []
    label_ = []   # 标签
    position = [] # 坐标
    for i in range(len(train_data)):
        featuretf_exp.append(train_data[i][0])
        featuretarget_exp.append(train_data[i][1])
        net_tf_s.append(train_data[i][2])
        net_tf_t.append(train_data[i][3])
        net_target_s.append(train_data[i][4])
        net_target_t.append(train_data[i][5])
        label_.append(train_data[i][6])
        position.append(train_data[i][7])

    featuretf_exp = np.array(featuretf_exp)
    featuretarget_exp = np.array(featuretarget_exp)
    net_tf_s = np.array(net_tf_s)
    net_tf_t = np.array(net_tf_t)
    net_target_s = np.array(net_target_s)
    net_target_t = np.array(net_target_t)
    # dataX = feature[:,np.newaxis,:,np.newaxis]  # (6500, 1, 48, 1)
    dataX_tf = featuretf_exp[:,np.newaxis,:]  # (6500, 1, 48) (6500, 1, 24)
    dataX_target = featuretarget_exp[:,np.newaxis,:]  # (6500, 1, 24)
    net_tf_s = net_tf_s[:,np.newaxis,:]  # (6500, 1, 64)
    net_tf_t = net_tf_t[:,np.newaxis,:]  # (6500, 1, 64)
    net_target_s = net_target_s[:,np.newaxis,:]  # (6500, 1, 64)
    net_target_t = net_target_t[:,np.newaxis,:]  # (6500, 1, 64)
    print("the shape of dataX_tf: ",dataX_tf.shape)
    print("the shape of dataX_target: ",dataX_target.shape)
    print("the shape of net_tf_s: ", net_tf_s.shape)

    label_ = np.array(label_)
    # print('label_:', label_)
    # 对标签 独热编码
    labelY = to_categorical(label_,3)
    # print('label_: ',label_)
    # print('labelY: ',labelY)

    position = np.array(position)

    # text_save('label_.csv', label_)
    # text_save('labelY.csv', labelY)

    return dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position

