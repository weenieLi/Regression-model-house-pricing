import pandas as pd

### 代码开始 ### (≈ 2 行代码)
df = None
df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1081/challenge-1-beijing.csv")

df.head()

# features = df['公交','写字楼','医院',,'商场','地铁','学校','建造时间','楼层','面积']
features = df[['公交','写字楼','医院','商场','地铁','学校','建造时间','楼层','面积']]
target = df['每平米价格']

pd.concat([features, target], axis=1).head()

split_num = int(len(df)*0.7) # 70% 分割数

### 代码开始 ### (≈ 4 行代码)
X_train = features[:split_num]
y_train = target[:split_num]
X_test = features[split_num:]
y_test = target[split_num:]
### 代码结束 ###

len(X_train), len(y_train), len(X_test), len(y_test)

from sklearn.linear_model import LinearRegression

### 代码开始 ### (≈ 2 行代码)
model = LinearRegression()
model.fit(X_train,y_train)

### 代码结束 ###

model.coef_[:3], len(model.coef_)

import numpy as np


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """

    ### 代码开始 ### (≈ 2 行代码)
    n = len(y_true)

    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    ### 代码结束 ###

    return mape

y_true = y_test.values
y_pred = model.predict(X_test)
mape(y_true, y_pred)

