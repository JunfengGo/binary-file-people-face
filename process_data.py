#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
from numpy import *
import pandas as pd
import readBinFile as rb
#预处理数据
#各个文件名
filename='feature.bin'
filename_test='feature_test.bin'
filename_label='label.txt'
#这个函数没用
def meanX(dataX):
    return mean (dataX, axis=0)
#pca 降维 2048->500维
def pca(XMat, k):
    average = meanX (XMat)
    m, n = shape (XMat)
    data_adjust = []
    avgs = tile (average, (m, 1))
    data_adjust = XMat - avgs
    covX = cov (data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = linalg.eig (covX)  # 求解协方差矩阵的特征值和特征向量
    index = argsort (-featValue)  # 按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
         selectVec = matrix (featVec.T[index[:k]])  # 所以这里需要进行转置
         finalData = data_adjust * selectVec.T
         reconData = (finalData * selectVec) + average
         return finalData
#预处理降维并且 regularization
def process_data_pca():
    col = 16032
    row = 2048
    data_array=rb.readDataFromBin(filename,col,row)
    data_after_PCA=pca(data_array,500)
    return auto_norm(data_after_PCA)
#同上
def process_test_data():
    col=18
    row=2048
    data_array=rb.readDataFromBin(filename_test,col,row)
    data_after_PCA = pca (data_array, 500)
    return auto_norm(data_after_PCA)
#得到标签
def get_label():
    label_num=[]
    label_name=[]
    f=open(filename_label)
    for line in f.readlines():
        line_block=line.split('  ')
        label_num.append(line_block[0])
        label_name.append(line_block[1])
    return label_num,label_name
#regularization处理函数
def auto_norm(data_matrix):
    m,n=shape(data_matrix)
    mat_data=matrix(zeros((m,n)))
    range=data_matrix.max(0)-data_matrix.min(0)
    data_matrix=(data_matrix-data_matrix.min(0))/range+mat_data
    return data_matrix
print(process_data_pca())

