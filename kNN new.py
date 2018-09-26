import numpy as np
import operator

def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]]) #四个特征坐标
    labels=['A','A','B','B'] #四个特征标签
    return group, labels

def classify0(inX,dataSet,labels,k): #inX: 测试集；dataSet：训练集；k：去的训练集个数即选择距离最小的几个点
    dataSetSize=dataSet.shape[0] #shape[0]取行数；shape[1]取列数
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet #重复inX，列方向上与dataSet相同，行方向上仅出现一次，再减去dataSet各点求x与y方向上的
    sqDiffMat=diffMat**2 #将上面的结果取平方
    sqDistance=sqDiffMat.sum(axis=1)  #将上面的平方后的结果按列求和
    distance=sqDistance**0.5 #再开方，可以得出4个距离
    sortedDistIndicies=distance.argsort() #上面输出的是一个1*4矩阵，argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。
    #例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    #x=np.array([1,4,3,-1,6,9])
    #y=array([3,0,2,1,4,5])
    classCount={}
    for i in range (k):
        voteIlabel=labels[sortedDistIndicies[i]]#取出前k个元素的类别 [i]指的是第几个
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #每拿到一次label就计次一次再+1，如果没有取出到那就返回0
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #item就是该字典中所有的项；itemgetter(1)按照字典的值进行排序，itemgetter(0)按照字典的字母顺序进行排列，reverse：降序）
        return sortedClassCount[0][0] ##取出返回次数最多的类别

if __name__ == '__main__': #在.py直接被运行时，代码块将直接被运行；当以模块导入时，if main之下的代码不会被运行
    group,labels=createDataSet()
    test=[101,20]
    test_class=classify0(test,group,labels,3)
    print(test_class)
