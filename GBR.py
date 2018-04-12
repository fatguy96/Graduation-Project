import numpy as np  # numpy库
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库

# 数据准备
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
data_ = np.array(data)
data = scale(data_)
data_scaled = StandardScaler().fit(data_)

shuffle_indices = np.random.permutation(np.arange(len(data)))
data_ = data[shuffle_indices]

train_size = int(0.7*len(data_))
X = data_[0:train_size, 3:14]
verify_x = data_[train_size:, 3:14]
y = data_[0:train_size, 0]
verify_y = data_[train_size:, 0]

# 训练回归模型
n_folds = 6  # 设置交叉检验的次数

model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象

model_names = ['SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_svr, model_gbr]  # 不同回归模型对象的集合

cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X, y, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X, y).predict(verify_x))  # 将回归训练中得到的预测y存入列表

# 模型效果指标评估
n_samples, n_features = X.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(2):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(verify_y, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('cross validation result:')  # 打印输出标题
print(df1)  # 打印输出交叉检验的数据框
print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线

# 模型效果可视化
plt.figure()  # 创建画布
plt.subplot(2, 1, 1)
plt.plot(np.arange(verify_x.shape[0]), verify_y, color='k', label='true y')  # 画出原始值的曲线
plt.title('real data')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real value')  # y轴标题

plt.subplot(2, 2, 3)
plt.plot(np.arange(verify_x.shape[0]), verify_y, color='k', label='true y')  # 画出原始值的曲线
plt.plot(np.arange(verify_x.shape[0]), pre_y_list[0], 'r', label="SVR")  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题

plt.subplot(2, 2, 4)
plt.plot(np.arange(verify_x.shape[0]), verify_y, color='k', label='true y')  # 画出原始值的曲线
plt.plot(np.arange(verify_x.shape[0]), pre_y_list[1], 'b', label="GBR")  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
