# 数据准备
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler

data_set_path = 'sample/K1135_20_allflow_version3.csv'
data = []
i = 1
with open(data_set_path, 'r') as f:
    for line in f:
        sample = line.strip().split(',')
        if len(sample) == 15 and i >= 577:
            data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                         float(sample[4]), float(sample[5]), float(sample[6]),
                         float(sample[7]), float(sample[8]), float(sample[9]),
                         float(sample[10]), float(sample[11]), float(sample[12]),
                         float(sample[13]), float(sample[14])])
        i += 1

# 规则化
data_ = np.array(data)
data = scale(data_)
data_scaled = StandardScaler().fit(data_)

train_size = int(0.7*len(data_))
verify_x = data[train_size:, 3:14]
verify_y = data[train_size:, 0]

with open('save/svr.pickle', 'rb') as f:
    model_svr = pickle.load(f)
with open('save/gbr.pickle', 'rb') as f:
    model_gbr = pickle.load(f)
with open('save/br.pickle', 'rb') as f:
    model_br = pickle.load(f)
with open('save/etc.pickle', 'rb') as f:
    model_etc = pickle.load(f)

pre_y_list = []
model_names = ['SVR', 'GBR', 'BR', 'ETC']  # 不同模型的名称列表
model_dic = [model_svr, model_gbr, model_br, model_etc]  # 不同回归模型对象的集合
for model in model_dic:  # 读出每个回归模型对象
    pre_y_list.append(model.predict(verify_x))  # 将回归训练中得到的预测y存入列表

num_epoch = int(len(verify_y)/288)

# 模型效果指标评估
n_samples, n_features = verify_x.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(4):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(verify_y, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框

print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线

for i in range(num_epoch):
    # 模型效果可视化
    plt.figure()  # 创建画布
    plt.subplot(5, 1, 1)
    plt.plot(np.arange(288), verify_y[i * 288: (i + 1) * 288], 'k.', label='true y')  # 画出原始值的曲线
    plt.title('real data')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real value')  # y轴标题

    plt.subplot(5, 1, 2)
    plt.plot(np.arange(288), verify_y[i * 288: (i + 1) * 288], 'k.', label='true y')  # 画出原始值的曲线
    plt.plot(np.arange(288), pre_y_list[0][i * 288: (i + 1) * 288], 'r.', label="SVR")  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题

    plt.subplot(5, 1, 3)
    plt.plot(np.arange(288), verify_y[i * 288: (i + 1) * 288], 'k.', label='true y')  # 画出原始值的曲线
    plt.plot(np.arange(288), pre_y_list[1][i * 288: (i + 1) * 288], 'b.', label="GBR")  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题

    plt.subplot(5, 1, 4)
    plt.plot(np.arange(288), verify_y[i * 288: (i + 1) * 288], 'k.', label='true y')  # 画出原始值的曲线
    plt.plot(np.arange(288), pre_y_list[2][i * 288: (i + 1) * 288], 'b.', label="BR")  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题

    plt.subplot(5, 1, 5)
    plt.plot(np.arange(288), verify_y[i * 288: (i + 1) * 288], 'k.', label='true y')  # 画出原始值的曲线
    plt.plot(np.arange(288), pre_y_list[3][i * 288: (i + 1) * 288], 'b.', label="ECT")  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题

    plt.savefig('/home/fate/Desktop/GBR_point/test%d.png' % (i + 1))
    plt.close()