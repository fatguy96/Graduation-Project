import numpy as np  # numpy库
import pickle

from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
import pandas as pd  # 导入pandas

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

# 规则化
data_ = np.array(data)
data = scale(data_)
data_scaled = StandardScaler().fit(data_)

train_size = int(0.7*len(data_))

# 打乱顺序
shuffle_indices = np.random.permutation(np.arange(len(data)))
data_ = data[shuffle_indices]

X = data_[0:train_size, 3:14]
y = data_[0:train_size, 0]

# 训练回归模型
n_folds = 6  # 设置交叉检验的次数


model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor(n_estimators=150)  # 建立梯度增强回归模型对象
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_names = ['SVR', 'GBR', 'BR', 'ETC']  # 不同模型的名称列表
model_dic = [model_svr, model_gbr, model_br, model_etc]  # 不同回归模型对象的集合

cv_score_list = []  # 交叉检验结果列表

for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X, y, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    model.fit(X, y)  # 将回归训练中得到的预测y存入列表


n_samples, n_features = X.shape  # 总样本量,总特征数
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框

print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('cross validation result:')  # 打印输出标题
print(df1)  # 打印输出交叉检验的数据框
print(70 * '-')  # 打印分隔线


with open('save/svr.pickle', 'wb') as f:
    pickle.dump(model_svr, f)

with open('save/gbr.pickle', 'wb') as f:
    pickle.dump(model_gbr, f)

with open('save/br.pickle', 'wb') as f:
    pickle.dump(model_br, f)

with open('save/etc.pickle', 'wb') as f:
    pickle.dump(model_etc, f)
