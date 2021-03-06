import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

#获取iris（鸢尾花）数据集
iris = datasets.load_iris()
#获取鸢尾花属性
iris_x = iris.data      
#获取鸢尾花已分类
iris_y = iris.target

#检查
#print(iris_x)
#print(iris_y)


#将数据分为测试集合训练集，测试集占比设置变量为test_size
x_train,x_test,y_train,y_test = train_test_split(iris_x,iris_y,test_size = 0.3)

#在数据分集的时候数据顺序也被打乱，在机器学习中乱序的数据（即各个类别杂糅的训练集）更利于分类
#print(y_train)


KNN = KNeighborsClassifier()
#K近邻训练
KNN.fit(x_train,y_train)

#预测
result = KNN.predict(x_test)
print("预测分类结果为：",result)
print("真实分类结果为：",y_test)

count = 0
i = 0
for j in result:
    if(result[i] == y_test[i]):
        count += 1
    i += 1
print("准确率:",count/(i+1))
  
