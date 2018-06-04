from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.model_selection import cross_val_score  # 交叉检验
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


class My_SVR:
    def __init__(self, data, cv):

        data = np.array(data)
        data_ = scale(data)
        data_scaled = StandardScaler().fit(data)

        train_size = int(0.7*len(data_))

        self.data_scaled = data_scaled
        self.train_x = data_[:train_size, 3:14]
        self.train_y = data_[:train_size, 0]
        self.verify_x = data_[train_size:, 3:14]
        self.verify_y = data_[train_size:, 0]
        self.n_folds = cv

    def train(self):
        model_svr = SVR(kernel='rbf', C=1e1, gamma=0.1)  # 建立支持向量机回归模型对象
        model_names = ['SVR']  # 不同模型的名称列表

        scores = cross_val_score(model_svr, self.train_x, self.train_y, cv=self.n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        model_svr.fit(self.train_x, self.train_y)  # 将回归训练中得到的预测y存入列表

        df1 = pd.DataFrame([scores], index=model_names)  # 建立交叉检验的数据框
        print(70 * '-')  # 打印分隔线
        print('cross validation result:')  # 打印输出标题
        print(df1)  # 打印输出交叉检验的数据框
        print(70 * '-')  # 打印分隔线

        with open('svr_model/svr.pickle', 'wb') as f:
            pickle.dump(model_svr, f)

    def verify(self):

        model_names = ['SVR']
        with open('svr_model/svr.pickle', 'rb') as f:
            model_svr = pickle.load(f)

        pre_y = model_svr.predict(self.verify_x)  # 将回归训练中得到的预测y存入列表

        model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
        model_metrics_list = []  # 回归评估指标列表

        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(self.verify_y, pre_y)  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表

        real_y = self.verify_y * self.data_scaled.scale_[0] + self.data_scaled.mean_[0]
        pre_y_r = pre_y * self.data_scaled.scale_[0] + self.data_scaled.mean_[0]

        acc = np.average(np.abs(real_y-pre_y_r)/real_y)
        tmp_list.append("{}%".format(round((acc*100), 2)))
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

        num_epoch = int(len(self.verify_x)/288)
        for i in range(num_epoch):
            # 模型效果可视化
            plt.figure()  # 创建画布
            plt.plot(np.arange(288), real_y[i * 288: (i + 1) * 288], 'k.', label='real')
            plt.plot(np.arange(288), pre_y_r[i * 288: (i + 1) * 288], 'b.', label='predict')
            plt.legend(loc='upper right')  # 图例位置
            plt.ylabel('flow')  # y轴标题
            plt.savefig('svr_test/test%d.png' % (i + 1))
            plt.close()

    def predict(self, predict_x):
        predict_x = (predict_x - self.data_scaled.mean_[3:14])/self.data_scaled.scale_[3:14]

        with open('svr_model/svr.pickle', 'rb') as f:
            model_svr = pickle.load(f)

        y = model_svr.predict(predict_x)

        y = y * self.data_scaled.scale_[0:1] + self.data_scaled.mean_[0:1]

        return y
