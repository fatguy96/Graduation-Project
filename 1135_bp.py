import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler

W1 = tf.Variable(tf.random_normal([11, 9], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([9], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([9, 3], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))
# W3 = tf.Variable(tf.random_normal([16, 16], stddev=1, seed=1, mean=0))
# b3 = tf.Variable(tf.random_normal([16], stddev=1, seed=1, mean=0))
# W4 = tf.Variable(tf.random_normal([16, 16], stddev=1, seed=1, mean=0))
# b4 = tf.Variable(tf.random_normal([16], stddev=1, seed=1, mean=0))
# W5 = tf.Variable(tf.random_normal([12, 3], stddev=1, seed=1, mean=0))
# b5 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))
# W6 = tf.Variable(tf.random_normal([16, 3], stddev=1, seed=1, mean=0))
# b6 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 11), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 3), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)

a1 = tf.nn.relu(z1, "a1")
y = tf.matmul(a1, W2)
# a2 = tf.nn.relu(z2, 'a2')
# z3 = tf.add(tf.matmul(a2, W3), b3)
# a3 = tf.nn.relu(z3, "a3")
# z4 = tf.add(tf.matmul(a3, W4), b4)
# a4 = tf.nn.relu(z4, "a4")
# y = tf.add(tf.matmul(a2, W5), b5)
# a5 = tf.nn.relu(z5, "a5")
# y = tf.add(tf.matmul(a5, W6), b6)

# loss = tf.reduce_mean(tf.square(y_ - y))
# train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
global_step = tf.Variable(0)
loss = tf.reduce_mean(tf.square(y_ - y))
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


def load_data(Type, shuffle=False):
    """

    :param Type: 选取各部分的数据集
    :param shuffle: 是否打乱
    :return: 返回相应的数据集
    """

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
    data_scaled = StandardScaler().fit(data_)

    # 打乱样本的顺序
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        data = data[shuffle_indices]

    train_example = data[:, 3:14]
    train_label = data[:, 0:3]

    # 提取训练样本(70%)和测试样本(30%)
    train_size = int(0.7 * len(data))

    if Type == "Train":
        return train_example[0:train_size], train_label[0:train_size], data_scaled
    elif Type == "Verify":
        return train_example[train_size:], train_label[train_size:], data_scaled
    else:
        return train_example, train_label, data_scaled


def train():

    org_x, org_y, _ = load_data("Train", False)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        plt.title("1135_flow_bp_ui_loss and test_epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        for e in range(15000):
            _, total_loss = sess.run([train_step, loss], feed_dict={x: org_x, y_: org_y})
            plt.plot(e, total_loss, 'r.')
            if e % 100 == 0:
                print('rate: {} epochs{}: {}'.format(learning_rate.eval(), e, total_loss))
        saver.save(sess, 'my_model/k1135_20_11')
        plt.show()


def predict(predict_input_x, my_scale):

    saver = tf.train.Saver()
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('my_model/')
        saver.restore(sess, model_file)
        predict_output_y = sess.run(y, feed_dict={x: predict_input_x})
        predict_output_y = predict_output_y * my_scale.scale_[0:3] + my_scale.mean_[0:3]

    return predict_output_y


def verify(verify_input__x, verify_input_y, my_scale):

    saver = tf.train.Saver()

    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('my_model/')
        saver.restore(sess, model_file)

        predict_y = sess.run(y, feed_dict={x: verify_input__x})
        # 模型效果指标评估
        model_names = ["BP"]
        model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
        model_metrics_list = []  # 回归评估指标列表
        for i in range(1):  # 循环每个模型索引
            tmp_list = []  # 每个内循环的临时结果列表
            for m in model_metrics_name:  # 循环每个指标对象
                tmp_score = m(verify_input_y, predict_y)  # 计算每个回归指标结果
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
        predict_y = predict_y * my_scale.scale_[0:3] + my_scale.mean_[0:3]
        verify_y = verify_input_y * my_scale.scale_[0:3] + my_scale.mean_[0:3]

        verify_y = verify_y[:, 0].reshape(-1, 1)
        predict_y = predict_y[:, 0].reshape(-1, 1)

        epoch = int(len(verify_y) / 288)
        for i in range(epoch):
            plt.figure()
            plt.xlabel("time")
            plt.ylabel("number")
            plt.plot(verify_y[i * 288: (i+1) * 288], 'b.', label="Actual value")
            plt.plot(predict_y[i * 288: (i+1) * 288], 'r.', label="Predictive value")
            plt.legend(loc='upper right')
            plt.savefig('verify_image/K1135_20_10/test%d.png' % (i + 1))
            plt.close()


if __name__ == '__main__':
    main_x, main_y, scale_ = load_data("Verify", False)
    choices = {'train': train, 'predict': predict, 'verify': verify}
    parser = argparse.ArgumentParser(description='train, predict and test bp')
    parser.add_argument('role', choices=choices, help='which role to play')
    parser.add_argument('-x', metavar='x', type=float, default=main_x, help='which x to predict')
    parser.add_argument('-y', metavar='y', type=float, default=main_y, help='which y to verify')
    parser.add_argument('-scale', metavar='scale', type=scale, default=scale_)
    args = parser.parse_args()
    func = choices[args.role]
    if args.role == 'train':
        func()
    elif args.role == 'predict':
        func(args.x, args.scale)
    elif args.role == 'verify':
        func(args.x, args.y, args.scale)
