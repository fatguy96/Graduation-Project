import tensorflow as tf
import numpy as np
import matplotlib.pyplot
from sklearn.preprocessing import scale, StandardScaler
from tkinter import Tk, Button, Frame, TOP, W, YES, X, BOTH, LEFT, PhotoImage, Label, filedialog, N, NW, NE, BOTTOM, E, \
    CENTER

# TODO: 选择合适的网络结构以及合适的输入单元
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


# TODO: 选择合适的优化算法和学习率
global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.square(y_ - y))
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 600, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


def load_data(style, filename='../sample/K1135_20_L34_version2.csv', shuffle=True):
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
    train_size = int(0.7*len(data))

    if style == "Train":
        return train_example[0:train_size], train_label[0:train_size], data_scaled

    elif style == "Verify":
        return train_example[train_size:], train_label[train_size:], data_scaled

    else:
        return train_example, train_label, data_scaled


def train(filename):

    train_x, train_y, data_scaled = load_data("Train", filename=filename, shuffle=True)

    names = filename.split('/')

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        matplotlib.pyplot.figure()
        matplotlib.pyplot.title("train_loss")
        matplotlib.pyplot.xlabel("epoch")
        matplotlib.pyplot.ylabel("loss")

        loss_list = []
        for e in range(20000):
            _, total_loss = sess.run([train_step, loss], feed_dict={x: train_x, y_: train_y})
            loss_list.append(total_loss)
            print('epochs{}: {}'.format(e+1, total_loss))

        matplotlib.pyplot.plot(loss_list, 'r')
        saver.save(sess, '../1135_flow_bp_ui_model/%s' % names[-1][0:-4])
        matplotlib.pyplot.savefig('../1135_flow_bp_ui_loss/{}.png'.format(names[-1][0:-4]))
        matplotlib.pyplot.close()
    return data_scaled


def predict(predict_input_x):

    saver = tf.train.Saver()

    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('../1135_flow_bp_ui_model/')

        saver.restore(sess, model_file)

        predict_output_y = sess.run(y, feed_dict={x: predict_input_x})

    return predict_output_y


if __name__ == '__main__':

    UI_filename = ""
    scaled_ = None

    def open_my():
        btn1['state'] = 'disabled'
        btn2['state'] = 'disabled'
        btn3['state'] = 'disabled'
        filename = filedialog.askopenfilename()
        filename = str(filename)
        if filename != "" and filename.endswith('.csv'):
            global UI_filename
            UI_filename = filename
            try:
                # 不知道为啥,没加这一句, 在plt.close的时候关闭了UI
                matplotlib.pyplot.figure()

                btn1['state'] = 'normal'

            except EOFError as e:
                print(e)
        else:
            pass


    def train_my():
        if UI_filename != "":
            try:
                global scaled_
                scaled_ = train(filename=UI_filename)
                names = UI_filename.split('/')
                imgInfo['file'] = '../1135_flow_bp_ui_loss/{}.png'.format(names[-1][0:-4])
                lab.configure(image=imgInfo)  # 重新设置Label图片
                btn2['state'] = 'normal'
                btn3['state'] = 'normal'
            except Exception as e:
                print(e)


    def verify_my():
        if UI_filename != "":
            try:
                verify_x, verify_y, data_scaled = load_data("Verify", filename=UI_filename,
                                                            shuffle=False)

                y_from_x = predict(verify_x)

                verify_y = verify_y*data_scaled.scale_[0:3]+data_scaled.mean_[0:3]

                verify_y = (verify_y[:, 0]).reshape(-1, 1)

                y_from_x = y_from_x*data_scaled.scale_[0:3] + data_scaled.mean_[0:3]
                y_from_x = (y_from_x[:, 0]).reshape(-1, 1)

                epoch = int(len(y_from_x) / 288)

                for i in range(epoch):
                    matplotlib.pyplot.figure()
                    matplotlib.pyplot.xlabel("time")
                    matplotlib.pyplot.ylabel("number")
                    matplotlib.pyplot.plot(verify_y[i * 288: (i+1) * 288], 'b', label="Actual value")
                    matplotlib.pyplot.plot(y_from_x[i * 288: (i+1) * 288], 'r', label="Predictive value")
                    matplotlib.pyplot.legend()
                    matplotlib.pyplot.savefig('../1135_flow_bp_ui_test/test%d.png' % (i + 1))
                    matplotlib.pyplot.close()

                show_num = np.random.randint(0, epoch)
                imgInfo['file'] = '../1135_flow_bp_ui_test/test%d.png' % show_num
                lab.configure(image=imgInfo)  # 重新设置Label图片
            except Exception as e:
                print(e)


    def getCurrentX():
        # TODO: 提取当前时间的一个X进行预测
        pass
        return 0


    def predict_my():
        need_x = getCurrentX()
        need_x = (need_x - scaled_.mean_[3:14])/scaled_.scale_[3:14]
        predictY = predict(need_x)
        predictY = predictY*scaled_.scale[0:3] + scaled_.mean_[0:3]
        lab_5_['text'] = str(predictY[0])
        lab_10_['text'] = str(predictY[1])
        lab_15_['text'] = str(predictY[2])

    root = Tk()

    root.title('流量训练与预测')

    left = Frame(root)

    btn = Button(left, text="choice file", state='normal', command=open_my, width=5)
    btn.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn1 = Button(left, text='train', state='disabled', command=train_my, width=5)
    btn1.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn2 = Button(left, text='verify', state='disabled', command=verify_my, width=5)
    btn2.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn3 = Button(left, text='predict', state='disabled', command=predict_my, width=5)
    btn3.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    left.pack(side=LEFT, fill=BOTH, expand=YES)

    right = Frame(root)

    imgInfo = PhotoImage(file="../loss/login.png")

    lab = Label(right, image=imgInfo)

    lab.pack(fill=BOTH, expand=YES, side=TOP)

    rightBottom = Frame(right)

    lab_notification = Label(rightBottom, text="result:->>")
    lab_notification.pack(side=LEFT, anchor=CENTER)
    lab_5 = Label(rightBottom, text='5 min:')
    lab_5.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_5_ = Label(rightBottom, text='---  辆')
    lab_5_.pack(side=LEFT, anchor=CENTER)

    lab_10 = Label(rightBottom, text='10 min:')
    lab_10.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_10_ = Label(rightBottom, text="---  辆")
    lab_10_.pack(side=LEFT, anchor=CENTER)

    lab_15 = Label(rightBottom, text='15 min:')
    lab_15.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_15_ = Label(rightBottom, text="---  辆")
    lab_15_.pack(side=LEFT, anchor=CENTER)

    rightBottom.pack(fill=BOTH, expand=YES, side=BOTTOM)

    right.pack(side=LEFT, fill=BOTH, expand=YES)
    root.mainloop()



