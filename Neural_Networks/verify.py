import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    :param Type: 选取各部分的数据集
    :return: 返回相应的数据集
    """
    data_set_path = '../sample/test.csv'
    data = []
    i = 1
    with open(data_set_path, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 14 and i >= 577:
                data.append([
                            float(sample[0]), float(sample[1]), float(sample[2]),
                            float(sample[3]), float(sample[4]), float(sample[5]),
                            float(sample[6]), float(sample[7]), float(sample[8]),
                            float(sample[9]), float(sample[10]), float(sample[11]),
                            float(sample[12]), float(sample[13])])
            i += 1
    data = np.array(data)
    train_example = data[:, 3:14]
    train_label = data[:, 0:3]
    return train_example, train_label


W1 = tf.Variable(tf.random_normal([11, 18], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([18], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([18, 6], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([6], stddev=1, seed=1, mean=0))
W3 = tf.Variable(tf.random_normal([6, 3], stddev=1, seed=1, mean=0))
b3 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 11), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 3), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1, "a1")
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.relu(z2)
y = tf.add(tf.matmul(a2, W3), b3)

saver = tf.train.Saver()

verify_x, real_y = load_data()
with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint('../my_model/')
    saver.restore(sess, model_file)
    predict_y = sess.run(y, feed_dict={x: verify_x})
    verify_y = (real_y[:, 0]).reshape(-1, 1)
    predict_y = (predict_y[:, 0]).reshape(-1, 1)
    epoch = int(verify_x.__len__() / 288)
    for i in range(epoch):
        plt.figure()
        plt.xlabel("time")
        plt.ylabel("number")
        plt.plot(verify_y[i * 288: (i + 1) * 288], 'b', label="Actual value")
        plt.plot(predict_y[i * 288: (i + 1) * 288], 'r', label="Predictive value")
        plt.legend()
        plt.savefig('/home/fate/Desktop/1135/test%d.png' % (i + 1))
        plt.close()