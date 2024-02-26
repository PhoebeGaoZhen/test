

# import os
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']="0"

import pandas as pd
import numpy as np
import random
# import tensorflow as tf
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
import time,os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score,accuracy_score,recall_score,precision_score,matthews_corrcoef
from detailGRN_final import three_utils_CNNC, CNNC

iteration = 10 # 5CV的迭代次数
num_negative = 6000
path_network_name_type = 'DATA\\traindataHuman\\final_GRN\\new_GRN_COVID_GEN_counts_genename.csv'
path_expression = 'DATA\\traindataHuman\\final_expression\\COVID_GEN_counts.csv'

# output_directory = '.\\output_directory\\'
output_directory = '.\\output_directory\\CNNC\\'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
# 保存一个网络10次5CV的AUC平均值、标准差
network_dict_name = 'covid'      # 不同网络
save_index_path = '.\\three_results\\CNNC\\'  # auc保存路径
if not os.path.isdir(save_index_path):
    os.makedirs(save_index_path)

# 读取数据  基因表达数据  和 基因名称列表
EXP_cold_raw = pd.read_csv(path_expression, sep='\,', header=None,engine='python')  # 2206*25
EXP_cold = EXP_cold_raw.loc[1:,1:] # 2205*24
EXP_cold = np.array(EXP_cold) # 2205*24
EXP_cold_new = np.zeros((EXP_cold.shape[0],EXP_cold.shape[1]))
for i in range(EXP_cold.shape[0]):
    for j in range(EXP_cold.shape[1]):
        EXP_cold_new[i][j] = float(EXP_cold[i][j])

genename = EXP_cold_raw.loc[1:,0] # 2205
genename = np.array(genename)
# print(EXP_cold)
# print(genename)

# 读取数据  已知调控关联，TF,target gene, types(activator, repressor, unknown)
Ecoli_GRN_known = pd.read_csv(path_network_name_type, sep='\,', header=None,engine='python')  # 4329*3
# print(Ecoli_GRN_known)


# 获取GRN矩阵，2205*2205   activator 3: 2070, repressor 2: 2034, activator+repressor 1: 225, not regulate 0
Ecoli_GRN, num_activator, num_repressor, num_unknown = three_utils_CNNC.get_GRN(Ecoli_GRN_known,genename)
# print(Ecoli_GRN.shape)
# print(num_activator, num_repressor, num_unknown)


# 构建Ecoli cold样本（expression ）可以产生2205*2205个样本 包含时序基因表达值
positive2_data, positive1_data, negative0_data, feature_size = three_utils_CNNC.create_samples_histogram2d(EXP_cold_new, Ecoli_GRN,num_negative)
# print(len(positive1_data))
# print(len(positive2_data))
# print(feature_size)
# print(len(positive3_data))
# print(len(negative0_data))

# 4. 10次5CV
network_dict = {"AUROC mean": 0,
                 "AUROC std": 0,
                 "Recall mean": 0,
                 "Recall std": 0,
                 "Precision mean": 0,
                 "Precision std": 0,
                 "F1 mean": 0,
                 "F1 std": 0,
                "MCC mean": 0,
                "MCC std": 0,
                "Acc mean": 0,
                "Acc std": 0}
all_network_dict = {"AUROC": 0,
                 "Recall": 0,
                 "Precision": 0,
                 "F1": 0,
                "MCC": 0,
                "Acc": 0}
kf = KFold(n_splits=5, shuffle=True)  # 初始化KFold
netavgAUROCs = []  # 存放一个网络10次5CV的平均AUC
netavgRecalls = []

netavgPrecisions = []
netavgF1s = []
netavgMCCs = []
netavgAccs = []
for ki in range(iteration):  # 10次5CV
    print('\n')
    print("\nthe {}th five-fold cross-validation..........\n".format(ki + 1))

    random.shuffle(negative0_data)

    alldata = np.vstack((positive2_data, positive1_data))
    alldata = np.vstack((alldata, negative0_data[0:num_negative]))  # 随机选择2171个未知样本作为负样本
    random.shuffle(alldata)  # 共 2070 + 2034 + 2046=6150个样本
    # 将全部样本进行转换得到样本 标签 坐标
    dataX, labelY, position,inputshape = three_utils_CNNC.transform_data_histogram2d(alldata)  # 获取特征和标签

    # 5CV
    AUROCs = []
    Recalls = []
    Precisions = []
    F1s = []
    MCCs = []
    Accs = []
    for train_index, test_index in kf.split(dataX, labelY):  # 调用split方法切分数据
        # 6.1 划分4:1训练集 测试集
        #  print('train_index:%s , test_index: %s ' %(train_index,test_index))
        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = labelY[train_index], labelY[test_index]
        # positionX, positionY = position[train_index], position[test_index]


        input_shape = (feature_size[0], feature_size[1],1)
        nb_classes = 3
        classifier = CNNC.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=True, patience=5)

        # 3.2 划分数据集
        (trainXX, testXX, trainYY, testYY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1,
                                                              shuffle=True)

        score_1, score_int = classifier.fit_5CV(trainXX, trainYY, testXX, testYY,testX)

        # 8. 计算性能评价指标
        # 1个网络1折的AUC
        # testY, score_1 : (1230,3)
        testY_int = np.argmax(testY, axis=1)

        cm = confusion_matrix(testY_int, score_int)
        conf_matrix = pd.DataFrame(cm)

        AUC = roc_auc_score(testY, score_1, multi_class='ovo')
        Recall = recall_score(testY_int, score_int, average='weighted')
        Precision = precision_score(testY_int, score_int, average='weighted')
        F1 = f1_score(testY_int, score_int, average='weighted')
        MCC = matthews_corrcoef(testY_int, score_int)
        ACC = accuracy_score(testY_int, score_int)
        # print('------Weighted------')
        # print('Weighted precision', precision_score(testY_int, score_int, average='weighted'))
        # print('Weighted recall', recall_score(testY_int, score_int, average='weighted'))
        # print('Weighted f1-score', f1_score(testY_int, score_int, average='weighted'))
        # print('------Macro------')
        # print('Macro precision', precision_score(testY_int, score_int, average='macro'))
        # print('Macro recall', recall_score(testY_int, score_int, average='macro'))
        # print('Macro f1-score', f1_score(testY_int, score_int, average='macro'))
        # print('------Micro------')
        # print('Micro precision', precision_score(testY_int, score_int, average='micro'))
        # print('Micro recall', recall_score(testY_int, score_int, average='micro'))
        # print('Micro f1-score', f1_score(testY_int, score_int, average='micro'))
        # precision_aupr, recall_aupr, _ = precision_recall_curve(testY_int, score_int)
        # AUPR = auc(recall_aupr, precision_aupr)

        # 1个网络5折的AUC
        AUROCs.append(AUC)
        Recalls.append(Recall)
        Precisions.append(Precision)
        F1s.append(F1)
        MCCs.append(MCC)
        Accs.append(ACC)
        print('一次五折交叉验证（1个网络5折的AUC）的AUC')
        # print('\nAUROCs:')
        print(AUROCs)
        # print('\n')

        # 一次五折交叉验证（1个网络5折的AUC）的平均AUC值
    avg_AUROC = np.mean(AUROCs)
    avg_Recalls = np.mean(Recalls)
    avg_Precisions = np.mean(Precisions)
    avg_F1s = np.mean(F1s)
    avg_MCCs = np.mean(MCCs)
    avg_Accs = np.mean(Accs)

    # 10次5CV的AUC值，有10个值
    netavgAUROCs.append(avg_AUROC)  # 10个AUC值，长度为10
    netavgRecalls.append(avg_Recalls)
    netavgPrecisions.append(avg_Precisions)
    netavgF1s.append(avg_F1s)
    netavgMCCs.append(avg_MCCs)
    netavgAccs.append(avg_Accs)
print('十次五折交叉验证的所有AUC值--------------------------------------------')
print(netavgAUROCs)

# 10次5CV的AUC平均值、标准差，有1个值
AUROC_mean = np.mean(netavgAUROCs)
AUROC_std = np.std(netavgAUROCs, ddof=1)
Recall_mean = np.mean(netavgRecalls)
Recall_std = np.std(netavgRecalls)
Precision_mean = np.mean(netavgPrecisions)
Precision_std = np.std(netavgPrecisions)
F1_mean = np.mean(netavgF1s)
F1_std = np.std(netavgF1s)
MCC_mean = np.mean(netavgMCCs)
MCC_std = np.std(netavgMCCs)
Acc_mean = np.mean(netavgAccs)
Acc_std = np.std(netavgAccs)

AUROC_mean = float('{:.4f}'.format(AUROC_mean))
AUROC_std = float('{:.4f}'.format(AUROC_std))
Recall_mean = float('{:.4f}'.format(Recall_mean))
Recall_std = float('{:.4f}'.format(Recall_std))
Precision_mean = float('{:.4f}'.format(Precision_mean))
Precision_std = float('{:.4f}'.format(Precision_std))
F1_mean = float('{:.4f}'.format(F1_mean))
F1_std = float('{:.4f}'.format(F1_std))
MCC_mean = float('{:.4f}'.format(MCC_mean))
MCC_std = float('{:.4f}'.format(MCC_std))
Acc_mean = float('{:.4f}'.format(Acc_mean))
Acc_std = float('{:.4f}'.format(Acc_std))

# print(network[net] + "10次5CV的AUC平均值为：%.4f" % AUROC_mean)
# print(network[net] + "10次5CV的AUC方差为：%.4f" % AUROC_var)
# print(network[net] + "10次5CV的AUC标准差为:%.4f" % AUROC_std)

# 将AUC的平均值标准差存到字典中，用于后续保存
network_dict["AUROC mean"] = AUROC_mean
network_dict["AUROC std"] = AUROC_std
network_dict["Recall mean"] = Recall_mean
network_dict["Recall std"] = Recall_std
network_dict["Precision mean"] = Precision_mean
network_dict["Precision std"] = Precision_std
network_dict["F1 mean"] = F1_mean
network_dict["F1 std"] = F1_std
network_dict["MCC mean"] = MCC_mean
network_dict["MCC std"] = MCC_std
network_dict["Acc mean"] = Acc_mean
network_dict["Acc std"] = Acc_std

# 保存一个网络10次5CV的AUC平均值、标准差

filename = open(save_index_path + network_dict_name + '.csv', 'w')
for k, v in network_dict.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()

# 保存一个网络10次5CV的AUC等指标
all_network_dict["AUROC"] = netavgAUROCs
all_network_dict["Recall"] = netavgRecalls
all_network_dict["Precision"] = netavgPrecisions
all_network_dict["F1"] = netavgF1s
all_network_dict["MCC"] = netavgMCCs
all_network_dict["Acc"] = netavgAccs

filename = open(save_index_path + network_dict_name+ 'all.csv', 'w')
for k, v in all_network_dict.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()
