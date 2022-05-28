import numpy as np

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

from matplotlib import pyplot as plt
# %matplotlib inline

plt.scatter(x, y)
plt.xlabel("Area")
plt.ylabel("Price")


def square_loss(x, y, w0, w1):
    loss = sum(np.square(y - (w0 + w1*x)))
    return loss


def f(x, w0, w1):
    y = w0 + w1 * x
    return y

def w_calculator(x, y):
    n = len(x)
    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w0, w1

w_calculator(x,y)

w0 = w_calculator(x, y)[0]
w1 = w_calculator(x, y)[1]

square_loss(x, y, w0, w1)

x_temp = np.linspace(50, 120, 100)  # 绘制直线生成的临时点

plt.scatter(x, y)
plt.plot(x_temp, x_temp*w1 + w0, 'r')

f(150, w0, w1)

from sklearn.linear_model import LinearRegression

# 定义线性回归模型
model = LinearRegression()
model.fit(x.reshape(len(x), 1), y)  # 训练, reshape 操作把数据处理成 fit 能接受的形状

# 得到模型拟合参数 w0 w1
print(model.intercept_, model.coef_)

def w_matrix(x, y):
    w = (x.T * x).I * x.T * y
    return w

x = np.matrix([[1, 56], [1, 72], [1, 69], [1, 88], [1, 102],
               [1, 86], [1, 76], [1, 79], [1, 94], [1, 74]])
y = np.matrix([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

w_matrix(x, y.reshape(10, 1))


import pandas as pd

df = pd.read_csv(
    "https://labfile.oss.aliyuncs.com/courses/1081/course-5-boston.csv")

df.head()

features = df[['crim','rm','lstat']]
features.describe()

target = df['medv']  # 目标值数据

split_num = int(len(features)*0.7)  # 得到 70% 位置

X_train = features[:split_num]  # 训练集特征
y_train = target[:split_num]  # 训练集目标

X_test = features[split_num:]  # 测试集特征
y_test = target[split_num:]  # 测试集目标

model = LinearRegression()  # 建立模型
model.fit(X_train, y_train)  # 训练模型
print(model.coef_, model.intercept_)  # 输出训练后的模型参数和截距项

def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae

def mse_value(y_true, y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse

mae = mae_value(y_test.values, preds)
mse = mse_value(y_test.values, preds)

print("MAE: ", mae)
print("MSE: ", mse)