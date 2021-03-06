from sklearn import datasets
from sklearn.linear_model import LinearRegression

#加载数据和属性
data = datasets.load_boston()
data_x = data.data
data_y = data.target


#检查数据
#print(data_x)
#print(data_y)

#训练
model = LinearRegression()
model.fit(data_x,data_y)

print("预测价格：",model.predict(data_x[:10,:]))
print("实际价格：",data_y[:10])
