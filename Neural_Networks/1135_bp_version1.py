import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

# TODO:  优化神经网络结构
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

loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)


def load_data(Type):
    """

    :param Type: 选取各部分的数据集
    :return: 返回相应的数据集
    """
    data_set_path = '../sample/K1135_20_L34_version2.csv'
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
    train_example = data[:, 3:14]
    train_label = data[:, 0:3]
    if Type == "Train":
        return train_example[0:19872], train_label[0:19872]
    elif Type == "Verify":
        return train_example[19872:25920], train_label[19872:25920]
    else:
        return train_example, train_label


all_x, all_y = load_data("All")
max_x = np.max(all_x)
min_x = np.min(all_x)
max_y = np.max(all_y)
min_y = np.min(all_y)


def max_min_normalization(x, max, min):
    return (x - min) / (max - min)


def reverse_normalization(x, max, min):
    return x*(max - min) + min


def train():

    org_x, org_y = load_data("Train")

    # 正则化
    train_x = max_min_normalization(org_x, max_x, min_x)
    train_y = max_min_normalization(org_y, max_y, min_y)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        plt.title("test_loss and test_epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        for e in range(20000):
            _, total_loss = sess.run([train_step, loss], feed_dict={x: train_x, y_: train_y})
            plt.plot(e, total_loss, 'r.')
            print('epochs{}: {}'.format(e, total_loss))
        saver.save(sess, '../my_model/version_k1135_20_L34')
        plt.show()
        print("W1: ", W1.eval(), "b1: ", b1.eval())
        print("W2: ", W2.eval(), "b2: ", b2.eval())
        print("W3: ", W3.eval(), "b2: ", b3.eval())


def predict(predict_input_x):

    predict_x = max_min_normalization(predict_input_x, max_x, min_x)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('../my_model/')
        saver.restore(sess, model_file)
        predict_output_y = sess.run(y, feed_dict={x: predict_x})
        predict_output_y = reverse_normalization(predict_output_y, max_y, min_y)

    return predict_output_y


def verify(verify_input__x, verify_input_y):

    verify_x = max_min_normalization(verify_input__x, max_x, min_x)
    saver = tf.train.Saver()
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('../my_model/')
        saver.restore(sess, model_file)
        predict_y = sess.run(y, feed_dict={x: verify_x})

        verify_y = (verify_input_y[:, 0]).reshape(-1, 1)
        # 有待商榷
        predict_y = reverse_normalization(predict_y, max_y, min_y)
        predict_y = (predict_y[:, 0]).reshape(-1, 1)

        epoch = int(verify_x.__len__() / 288)
        for i in range(epoch):
            plt.figure()
            plt.xlabel("time")
            plt.ylabel("number")
            plt.plot(verify_y[i * 288: (i+1) * 288], 'b', label="Actual value")
            plt.plot(predict_y[i * 288: (i+1) * 288], 'r', label="Predictive value")
            plt.legend()
            plt.savefig('../verify_image/K1135_20_L34/test%d.png' % (i + 1))
            plt.close()


if __name__ == '__main__':
    main_x, main_y = load_data("Verify")
    choices = {'train': train, 'predict': predict, 'verify': verify}
    parser = argparse.ArgumentParser(description='train, predict and test bp')
    parser.add_argument('role', choices=choices, help='which role to play')
    parser.add_argument('-x', metavar='x', type=float, default=main_x, help='which x to predict')
    parser.add_argument('-y', metavar='y', type=float, default=main_y, help='which y to verify')
    args = parser.parse_args()
    func = choices[args.role]
    if args.role == 'train':
        func()
    elif args.role == 'predict':
        func(args.x)
    elif args.role == 'verify':
        func(args.x, args.y)
