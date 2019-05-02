from numpy import *
import operator

#生成训练集及类别
def createdataset():
    group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ["A","A","B","B"]
    return group,labels

#KNN分类函数
def classify(input_c,dataset,label,k):
    datasize = dataset.shape[0]

    #计算欧式距离
    diff = tile(input_c,(datasize,1)) - dataset
    sqdiff = diff**2
    squardist = sum(sqdiff,axis = 1)#行向量分别相加，从而得到新的一个行向量
    dist = squardist ** 0.5


    #对距离进行排序
    sorteddistindex = argsort(dist)
    classcount = {}
    for i in range(k):
        votelabel = label[sorteddistindex[i]]
        #对选取的K个样本所属的类别个数进行统计
        classcount[votelabel] = classcount.get(votelabel,0) + 1
    #选取出现的类别次数最多的类别
    maxcount = 0
    for key,vaule in classcount.items():
        if vaule > maxcount:
            maxcount = vaule
            classes = key

    return classes



dataset,label = createdataset()
input_c = array([1.1,0.3])
#input_c = input("请输入一个1*2的数组：")
k = 3

output = classify(input_c,dataset,label,k)
print("测试数据为:",input_c)
print("分类结果为:",output)
