import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse


# TODO:  优化神经网络
W1 = tf.Variable(tf.random_normal([9, 18], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([18], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([18, 6], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([6], stddev=1, seed=1, mean=0))
W3 = tf.Variable(tf.random_normal([6, 3], stddev=1, seed=1, mean=0))
b3 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 9), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 3), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1, "a1")
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.relu(z2)
y = tf.add(tf.matmul(a2, W3), b3)

loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)


def load_data():

    data_set_path = 'sample/K1135_20_L31_version1.csv'
    train_example = []
    train_label = []
    i = 1
    with open(data_set_path, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 13 and i >= 577:
                train_example.append([int(sample[4]), int(sample[5]), int(sample[6]),
                                      int(sample[7]), int(sample[8]), int(sample[9]),
                                      int(sample[10]), int(sample[11]), int(sample[12])])
                train_label.append([int(sample[1]), int(sample[2]), int(sample[3])])
            i += 1
    return np.array(train_example), np.array(train_label)


def max_min_normalization(x, max, min):
    return (x - min) / (max - min)


def reverse_normalization(x, max, min):
    return x*(max - min) + min


def train():

    org_x, org_y = load_data()

    # 正则化
    train_x = max_min_normalization(org_x, np.max(org_x), np.min(org_x))
    train_y = max_min_normalization(org_y, np.max(org_y), np.min(org_y))

    # shuffle
    # shuffle_indices = np.random.permutation(np.arange(len(train_y)))
    # train_x = train_x[shuffle_indices]
    # train_y = train_y[shuffle_indices]

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        plt.title("test_loss and test_epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        for e in range(10000):
            _, total_loss = sess.run([train_step, loss], feed_dict={x: train_x, y_: train_y})
            plt.plot(e, total_loss, 'r.')
            print('epochs{}: {}'.format(e, total_loss))
        saver.save(sess, 'my_model/version_one')
        plt.show()
        print("W1: ", W1.eval(), "b1: ", b1.eval())
        print("W2: ", W2.eval(), "b2: ", b2.eval())
        print("W3: ", W3.eval(), "b2: ", b3.eval())


def predict(predict_x):

    org_x, org_y = load_data()
    train_x = max_min_normalization(org_x, np.max(org_x), np.min(org_x))
    train_y = max_min_normalization(org_y, np.max(org_y), np.min(org_y))

    saver = tf.train.Saver()
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('my_model/')
        saver.restore(sess, model_file)

        x_org = train_x[14976:15264, :]
        y_org = train_y[14976:15264, :]

        x = tf.placeholder(tf.float32, shape=(None, 9), name='example')
        z1 = tf.add(tf.matmul(x, sess.run(W1)), b1)
        a1 = tf.nn.relu(z1, "a1")
        z2 = tf.add(tf.matmul(a1, W2), b2)
        a2 = tf.nn.relu(z2)
        y = tf.add(tf.matmul(a2, W3), b3)
        predict_y = sess.run(y, feed_dict={x: x_org})

        y_org = reverse_normalization(y_org, np.max(org_y), np.min(org_y))
        y_org = (y_org[:, 0]).reshape(-1, 1)
        plt.plot(y_org, 'b.')

        predict_y = reverse_normalization(predict_y, np.max(org_y), np.min(org_y))
        predict_y = (predict_y[:, 0]).reshape(-1, 1)
        plt.plot(predict_y, 'r.')

        plt.show()


if __name__ == '__main__':

    org_x, org_y = load_data()
    # 正则化
    train_x = max_min_normalization(org_x, np.max(org_x), np.min(org_x))
    x_org = train_x[14976:15264, :]

    choices = {'train': train, 'predict': predict}
    parser = argparse.ArgumentParser(description='train and test_model bp')
    parser.add_argument('role', choices=choices, help='which role to play')
    parser.add_argument('-x', metavar='x', type=int, default=x_org, help='which x to predict')
    args = parser.parse_args()
    func = choices[args.role]
    if args.role == 'train':
        func()
    else:
        func(args.x)


