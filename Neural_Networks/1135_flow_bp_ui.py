import tkinter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

# TODO:  优化神经网络结构
# W1 = tf.Variable(tf.random_normal([11, 18], stddev=1, seed=1, mean=0))
# b1 = tf.Variable(tf.random_normal([18], stddev=1, seed=1, mean=0))
# W2 = tf.Variable(tf.random_normal([18, 15], stddev=1, seed=1, mean=0))
# b2 = tf.Variable(tf.random_normal([15], stddev=1, seed=1, mean=0))
# W3 = tf.Variable(tf.random_normal([15, 12], stddev=1, seed=1, mean=0))
# b3 = tf.Variable(tf.random_normal([12], stddev=1, seed=1, mean=0))
# W4 = tf.Variable(tf.random_normal([12, 9], stddev=1, seed=1, mean=0))
# b4 = tf.Variable(tf.random_normal([9], stddev=1, seed=1, mean=0))
# W5 = tf.Variable(tf.random_normal([9, 6], stddev=1, seed=1, mean=0))
# b5 = tf.Variable(tf.random_normal([6], stddev=1, seed=1, mean=0))
# W6 = tf.Variable(tf.random_normal([6, 3], stddev=1, seed=1, mean=0))
# b6 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))
#
#
# x = tf.placeholder(tf.float32, shape=(None, 11), name='example')
# y_ = tf.placeholder(tf.float32, shape=(None, 3), name='label')
#
# z1 = tf.add(tf.matmul(x, W1), b1)
# a1 = tf.nn.relu(z1, "a1")
# z2 = tf.add(tf.matmul(a1, W2), b2)
# a2 = tf.nn.relu(z2)
# z3 = tf.add(tf.matmul(a2, W3), b3)
# a3 = tf.nn.relu(z3)
# z4 = tf.add(tf.matmul(a3, W4), b4)
# a4 = tf.nn.relu(z4)
# z5 = tf.add(tf.matmul(a4, W5), b5)
# a5 = tf.nn.relu(z5)
# y = tf.add(tf.matmul(a5, W6), b6)
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

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.square(y_ - y))
starter_learning_rate = 0.0005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 300, 0.9, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


def load_data(style, filename='../sample/K1135_20_L34_version2.csv'):
    """

    :param style: 选取各部分的数据集
    :param filename: 选取的数据集
    :return: 返回相应的数据集
    """
    data = []
    i = 1
    with open(filename, 'r') as f:
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
    if style == "Train":
        return train_example[0:19872], train_label[0:19872]
    elif style == "Verify":
        return train_example[19872:], train_label[19872:]
    else:
        return train_example, train_label


def max_min_normalization(x, max, min):
    return (x - min) / (max - min)


def reverse_normalization(x, max, min):
    return x*(max - min) + min


def train(filename):

    org_x, org_y = load_data("Train", filename=filename)
    all_x, all_y = load_data("All", filename=filename)

    names = filename.split('/')

    max_x = np.max(all_x)
    min_x = np.min(all_x)
    max_y = np.max(all_y)
    min_y = np.min(all_y)

    # 正则化
    train_x = max_min_normalization(org_x, max_x, min_x)
    train_y = max_min_normalization(org_y, max_y, min_y)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        plt.title("test_loss and test_epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        for e in range(15000):
            _, total_loss = sess.run([train_step, loss], feed_dict={x: train_x, y_: train_y})
            plt.plot(e, total_loss, 'r.')
            print('epochs{}: {}'.format(e, total_loss))
        saver.save(sess, '../my_model/%s' % names[-1][0:-4])
        plt.legend()
        plt.savefig('../loss/%s' % names[-1][0:-4])
        plt.close()


def predict(predict_input_x, max_x, min_x, max_y, min_y):

    predict_x = max_min_normalization(predict_input_x, max_x, min_x)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('../my_model/')
        saver.restore(sess, model_file)
        predict_output_y = sess.run(y, feed_dict={x: predict_x})
        predict_output_y = reverse_normalization(predict_output_y, max_y, min_y)

    return predict_output_y


def verify(verify_input__x, verify_input_y, max_x, min_x, max_y, min_y):

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
            plt.savefig('/home/fate/Desktop/1135/test%d.png' % (i + 1))
            plt.close()


if __name__ == '__main__':

    UI_filename = ""
    max_x = min_x = max_y = min_y = 0

    def open_my():
        filename = tkinter.filedialog.askopenfilename()
        if filename != "" and filename.endswith('.csv'):
            global UI_filename, max_x, min_x, max_y, min_y
            UI_filename = filename
            try:
                all_x, all_y = load_data("All", filename=UI_filename)
                max_x = np.max(all_x)
                min_x = np.min(all_x)
                max_y = np.max(all_y)
                min_y = np.min(all_y)
            except EOFError as e:
                print(e.__context__)
        else:
            pass


    def train_my():
        if UI_filename != "":
            try:
                train(filename=UI_filename)
            except Exception as e:
                print(e.__context__)


    def verify_my():
        if UI_filename != "":
            try:
                main_x, main_y = load_data('Verify', filename=UI_filename)
                verify(main_x, main_y, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
            except Exception as e:
                print(e.__context__)


    root = Tk()

    btn = Button(root, text="choice the csv to train or verify", command=open_my).pack(side=LEFT, expand=YES
                                                                                       , fill=BOTH)
    Button(root, text='train', command=train_my).pack(side=LEFT, expand=YES, fill=BOTH)

    Button(root, text='verify', command=verify_my).pack(side=LEFT, expand=YES, fill=BOTH)

    root.mainloop()
