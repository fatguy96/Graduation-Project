import random
from GA_bp.GA import GA
from LSTM.lstm import My_LSTM
from My_SVR.svr import My_SVR
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.model_selection import cross_val_score  # 交叉检验
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def load_data(filename='data/data.csv'):
    data = []
    i = 1
    with open(filename, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 15 and i >= 577:
                data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                            float(sample[4]), float(sample[5]), float(sample[6]),
                            float(sample[7]), float(sample[8]), float(sample[9]),
                            float(sample[10]), float(sample[11]), float(sample[12]),
                            float(sample[13]), float(sample[14])])
            i += 1
    data = np.array(data)
    return data


data = load_data()
train_size = int(0.7 * len(data))
column1 = data[:train_size, 3:14]
column2 = data[:train_size, 0:3]
column1_v = data[train_size:, 3:14]
column2_v = data[train_size:, 0:3]
data_ = np.array(data)
data_my = scale(data_)
data_scaled = StandardScaler().fit(data_)

svr = My_SVR(data, cv=6)

lstm = My_LSTM(rnn_unit=10, input_size=11, time_step=20, output_size=3, lr=0.006, data=data)

a = GA(30, 50, 138, 0.75, 0.05, column1, column2, column1_v, column2_v)

svr_pre_y = svr.predict(data[train_size:, 3:14])
ga_y = a.predict(data[train_size:, 3:14])

range_int = random.random()
if range_int > 0.48:
    pre_y = svr_pre_y
else:
    pre_y = ga_y[:, 0]

acc = np.average(np.abs(data[train_size:, 0] - pre_y)/data[train_size:, 0])

pre_y = (pre_y-data_scaled.mean_[0])/data_scaled.scale_[0]

model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表

tmp_list = []  # 每个内循环的临时结果列表
for m in model_metrics_name:  # 循环每个指标对象
    tmp_score = m(data_my[train_size:, 0], pre_y)  # 计算每个回归指标结果
    tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表


tmp_list.append("{}%".format(round((acc * 100), 2)))
model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

df2 = pd.DataFrame(model_metrics_list, index=['混合'], columns=['ev', 'mae', 'mse', 'r2', 'acc'])  # 建立回归指标的数据框

print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print(' ev \t\t\t  explained_variance')
print('mae \t\t\t  mean_absolute_error')
print('mse \t\t\t  mean_squared_error')
print(' r2 \t\t\t  r2')
print('acc \t\t\t  相对误差')
print(70 * '-')  # 打印分隔线


