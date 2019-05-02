from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#使用normalization归一化解决梯度弥散
#a = np.array([[10,2.7,3.6],[-100,5,-2],[120,20,40]])
#print(a)
#print(preprocessing.scale(a))

#n_informative是聚类簇数量，n_redundant是冗余特征的数量，n_clusters_per_class是每个类的簇数
#random_state确定数据集创建的随机数，scale乘以features移位features
#生成数据集
x,y = make_classification(n_samples = 300,n_features = 2,n_redundant = 0,n_informative = 2,
                          random_state = 22,n_clusters_per_class = 1,scale = 100)



#plt.scatter(x[:,0],x[:,1],c = y)
#plt.show()
#print(x)
#print(y)
#归一化数据
x = preprocessing.scale(x)
#分离训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
#SVC训练
clf = SVC()
clf.fit(x_train,y_train)
#测试并打分
print(clf.score(x_test,y_test))
