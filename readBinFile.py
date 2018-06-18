#读取二进制文件
import numpy as np
import struct
def readDataFromBin(filename, rows, cols):  
    f = open(filename, "rb")  
    pic = np.zeros((rows, cols))  
    for i in range(rows):  
        for j in range(cols):  
            data = f.read(4)  
            elem = struct.unpack("f", data)[0]
            pic[i][j] = elem  
    f.close()  
    return pic  
#
#
#data = readDataFromBin('/Users/guojunfeng/Desktop/machine_learn/feature.bin', 16032, 2048)