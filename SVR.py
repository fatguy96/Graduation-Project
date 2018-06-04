
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.model_selection import cross_val_score  # 交叉检验
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# plt.figure()
# # 加载数据
# data_set_path = 'data'
# data = []
# i = 1
# with open(data_set_path, 'r') as f:
#     for line in f:
#         sample = line.strip().split(',')
#         if len(sample) == 15 and i >= 577:
#             data.append([float(sample[1]), float(sample[2]), float(sample[3]),
#                          float(sample[4]), float(sample[5]), float(sample[6]),
#                          float(sample[7]), float(sample[8]), float(sample[9]),
#                          float(sample[10]), float(sample[11]), float(sample[12])])
#         i += 1
# data = np.array(data)
# data_ = np.array(data)
# data = scale(data_)
#
# data_predict = data
#
# data_scaled = StandardScaler().fit(data_)
#
# shuffle_indices = np.random.permutation(np.arange(len(data)))
# data = data[shuffle_indices]
#
# train_size = int(0.7 * len(data))
#
# svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
# train_sizes, train_scores_svr, test_scores_svr = \
#     learning_curve(svr, data[0:train_size, 3:12], data[0:train_size, 0], train_sizes=np.linspace(0.1, 1, 10),
#                    scoring="neg_mean_squared_error", cv=10)
#
# plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
#          label="SVR")
#
# plt.xlabel("Train size")
# plt.ylabel("Mean Squared Error")
# plt.title('Learning curves')
# plt.legend(loc="best")
# plt.show()


def train(train_x, train_y):
    model_svr = SVR(kernel='rbf', C=1e1, gamma=0.1)  # 建立支持向量机回归模型对象
    model_names = ['SVR']  # 不同模型的名称列表

    scores = cross_val_score(model_svr, train_x, train_y, cv=6)  # 将每个回归模型导入交叉检验模型中做训练检验
    model_svr.fit(train_x, train_y)  # 将回归训练中得到的预测y存入列表

    df1 = pd.DataFrame([scores], index=model_names)  # 建立交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('cross validation result:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线

    with open('svr.pickle', 'wb') as f:
        pickle.dump(model_svr, f)


def verify(verify_x, verify_y, data_scaled):
    model_names = ['SVR']
    with open('svr.pickle', 'rb') as f:
        model_svr = pickle.load(f)

    pre_y = model_svr.predict(verify_x)  # 将回归训练中得到的预测y存入列表

    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表

    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(verify_y, pre_y)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表

    real_y = verify_y * data_scaled.scale_[0] + data_scaled.mean_[0]
    pre_y = pre_y * data_scaled.scale_[0] + data_scaled.mean_[0]

    acc = np.average(np.abs(real_y - pre_y) / real_y)
    tmp_list.append("{}%".format(acc * 100))
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2', 'acc'])  # 建立回归指标的数据框

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


def load_data(filename='data.csv'):
    data = []
    i = 1
    with open(filename, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if i >= 577:
                data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                            float(sample[4]), float(sample[5]), float(sample[6]),
                            float(sample[7]), float(sample[8]), float(sample[9]),
                            float(sample[10]), float(sample[11]), float(sample[12])])
            i += 1
    data = np.array(data)
    return data


if __name__ == '__main__':

    data = load_data()
    train_size = int(0.7 * len(data))
    data_ = np.array(data)

    data_my = scale(data_)
    data_scaled = StandardScaler().fit(data_)
    train(data_my[:train_size, 3:12], data_my[:train_size, 0])
    verify(data_my[train_size:, 3:12], data_my[train_size:, 0], data_scaled=data_scaled)