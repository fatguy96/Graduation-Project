import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def load_data():
    """
    :param Type: 选取各部分的数据集
    :return: 返回相应的数据集
    """
    data_set_path = '../sample/bp_test.csv'
    data = []
    i = 1
    with open(data_set_path, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 14:
                data.append([
                            float(sample[0]), float(sample[1]), float(sample[2]),
                            float(sample[3]), float(sample[4]), float(sample[5]),
                            float(sample[6]), float(sample[7]), float(sample[8]),
                            float(sample[9]), float(sample[10]), float(sample[11]),
                            float(sample[12]), float(sample[13])])
            i += 1
    data = np.array(data)
    train_example = data[:, 3:14]
    train_label = data[:, 0:1]
    return train_example, train_label


W1 = tf.Variable(tf.random_normal([11, 9], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([9], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([9, 3], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))
W3 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1, mean=0))
b3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 11), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1, "a1")
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.relu(z2)
y = tf.add(tf.matmul(a2, W3), b3)

saver = tf.train.Saver()

verify_x, real_y = load_data()
with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint('my_model/')
    saver.restore(sess, model_file)
    predict_y = sess.run(y, feed_dict={x: verify_x})

    # 模型效果指标评估
    model_names = ["BP"]
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    for i in range(1):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(real_y, predict_y)  # 计算每个回归指标结果
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

    verify_y = real_y.reshape(-1, 1)
    predict_y = predict_y.reshape(-1, 1)
    epoch = int(verify_x.__len__() / 288)
    for i in range(epoch):
        plt.figure()
        plt.xlabel("time")
        plt.ylabel("number")
        plt.plot(verify_y[i * 288: (i + 1) * 288], 'b', label="Actual value")
        plt.plot(predict_y[i * 288: (i + 1) * 288], 'r', label="Predictive value")
        plt.legend(loc='upper right')
        plt.savefig('/home/fate/Desktop/1135/test%d.png' % (i + 1))
        plt.close()
