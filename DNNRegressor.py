import tensorflow as tf
import numpy as np


# 数据准备
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

feature = data_[0:train_size, 3:14]
label = data_[0:train_size, 0]

feature_columns = [tf.contrib.layers.real_valued_column("")]
classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                           hidden_units=[15],
                                           optimizer=tf.train.RMSPropOptimizer(learning_rate=.001),
                                           activation_fn=tf.nn.relu)


classifier.fit(x=feature, y=label, max_steps=10000)

print(classifier.evaluate(x=feature, y=label))
