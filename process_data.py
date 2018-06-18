#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
from numpy import *
import pandas as pd
import readBinFile as rb
#Ԥ��������
#�����ļ���
filename='feature.bin'
filename_test='feature_test.bin'
filename_label='label.txt'
#�������û��
def meanX(dataX):
    return mean (dataX, axis=0)
#pca ��ά 2048->500ά
def pca(XMat, k):
    average = meanX (XMat)
    m, n = shape (XMat)
    data_adjust = []
    avgs = tile (average, (m, 1))
    data_adjust = XMat - avgs
    covX = cov (data_adjust.T)  # ����Э�������
    featValue, featVec = linalg.eig (covX)  # ���Э������������ֵ����������
    index = argsort (-featValue)  # ����featValue���дӴ�С����
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # ע����������ʱ����������numpy�Ķ�ά����(����)a[m][n]�У�a[1]��ʾ��1��ֵ
         selectVec = matrix (featVec.T[index[:k]])  # ����������Ҫ����ת��
         finalData = data_adjust * selectVec.T
         reconData = (finalData * selectVec) + average
         return finalData
#Ԥ����ά���� regularization
def process_data_pca():
    col = 16032
    row = 2048
    data_array=rb.readDataFromBin(filename,col,row)
    data_after_PCA=pca(data_array,500)
    return auto_norm(data_after_PCA)
#ͬ��
def process_test_data():
    col=18
    row=2048
    data_array=rb.readDataFromBin(filename_test,col,row)
    data_after_PCA = pca (data_array, 500)
    return auto_norm(data_after_PCA)
#�õ���ǩ
def get_label():
    label_num=[]
    label_name=[]
    f=open(filename_label)
    for line in f.readlines():
        line_block=line.split('  ')
        label_num.append(line_block[0])
        label_name.append(line_block[1])
    return label_num,label_name
#regularization������
def auto_norm(data_matrix):
    m,n=shape(data_matrix)
    mat_data=matrix(zeros((m,n)))
    range=data_matrix.max(0)-data_matrix.min(0)
    data_matrix=(data_matrix-data_matrix.min(0))/range+mat_data
    return data_matrix
print(process_data_pca())

