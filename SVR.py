from sklearn.model_selection import learning_curve
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
# 加载数据
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
data = np.array(data)
data_ = np.array(data)
data = scale(data_)

data_predict = data

data_scaled = StandardScaler().fit(data_)

shuffle_indices = np.random.permutation(np.arange(len(data)))
data = data[shuffle_indices]

train_size = int(0.7 * len(data))

svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, data[0:train_size, 3:14], data[0:train_size, 0], train_sizes=np.linspace(0.1, 1, 10),
                   scoring="neg_mean_squared_error", cv=10)

plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
         label="SVR")

plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
plt.show()
