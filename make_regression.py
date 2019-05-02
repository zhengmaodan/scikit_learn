from sklearn import datasets
import matplotlib.pyplot as plt

#其中，n_samples是数据量，n_features是特征量，n_targets是输出，noise是噪声
x,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=5)

plt.scatter(x,y)
plt.show()
