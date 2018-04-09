import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# TODO：根据后期的确定的样本进行修改 现在的样本为前两天当时的数据来确定今天当时的数据
def load_data(data_set_path):
    train_example = []
    train_label = []
    with open(data_set_path, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 3:
                train_example.append([int(sample[1]), int(sample[2])])
                train_label.append([int(sample[0])])
    return np.array(train_example), np.array(train_label)


def max_min_normalization(x, max, min):
    return (x - min) / (max - min)


data_path = 'test.csv'
train_x, train_y = load_data(data_path)

# 正则化
train_x = max_min_normalization(train_x, np.max(train_x), np.min(train_x))
train_y = max_min_normalization(train_y, np.max(train_y), np.min(train_y))

# shuffle
shuffle_indices = np.random.permutation(np.arange(len(train_y)))
train_x = train_x[shuffle_indices]
train_y = train_y[shuffle_indices]

batch_size = 64
num_batch = len(train_y)//batch_size
print(num_batch)

# TODO: 1、优化神经网络，模型就是一个单层神经网络
# TODO：3、考虑正则化的含义
W1 = tf.Variable(tf.random_normal([2, 15], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([15], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([15, 6], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([6], stddev=1, seed=1, mean=0))
W3 = tf.Variable(tf.random_normal([6, 1], stddev=1, seed=1, mean=0))
b3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 2), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1, "a1")
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.relu(z2)
y = tf.add(tf.matmul(a2, W3), b3)

loss = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    plt.title("1135_flow_bp_ui_loss and test_epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    for e in range(2000):
        total_loss = 0
        _, total_loss = sess.run([train_step, loss], feed_dict={x: train_x, y_: train_y})
        plt.plot(e, total_loss, 'r.')
        print('epochs{}: {}'.format(e, total_loss))
    plt.show()
    print("W1: ", W1.eval(), "b1: ", b1.eval())
    print("W2: ", W2.eval(), "b2: ", b2.eval())




