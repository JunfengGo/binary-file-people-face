#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
from numpy import *
import process_data as pr
from collections import Counter
train_data=pr.process_data_pca()
test_data=pr.process_test_data()
label_num,label_name=pr.get_label()
#计算相似度
def calculate_dis(k):
   m,n=shape(test_data)
   a,b=shape(train_data)
   for i in range(m):
       t=[]
       label = []
       label1 = []
       for v in range(a):
         total_distance=sum(power((test_data[i,:]-train_data[v,:]),2),1)
         t.append(float(real(total_distance)))
       t=array(t)
       order_dis=t.argsort()
       order_dis=order_dis[0:k]
       for item in order_dis:
        label.append (label_name[item])
        label1.append (label_num[item])
       #打印出最常出现的
       print (Counter (label).most_common (1))
       print (Counter (label1).most_common (1))
calculate_dis(10)
