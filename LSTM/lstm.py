# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # 定义常量
# rnn_unit = 10       # hidden layer units
# input_size = 11
# time_step = 20
# output_size = 3
# lr = 0.0006         # 学习率
# # ——————————————————导入数据——————————————————————
# f = open('sample/lstm_data.csv')
# df = pd.read_csv(f)     # 读入数据
# data = df.iloc[:, 1:15].values  # 取第1-12列
# train_size = int(len(data)*0.7)
#
#
# # 获取训练集
# def get_train_data(batch_size=60, train_end=train_size):
#     batch_index = []
#     data_train = data[:train_end]
#     normalized_train_data = (data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)  # 标准化
#     train_x, train_y = [], []   # 训练集
#     for i in range(len(normalized_train_data)-time_step):
#         if i % batch_size == 0:
#             batch_index.append(i)
#         x = normalized_train_data[i:i+time_step, :11]
#         y = normalized_train_data[i:i+time_step, 11:14]
#         train_x.append(x.tolist())
#         train_y.append(y.tolist())
#     batch_index.append((len(normalized_train_data)-time_step))
#     return batch_index, train_x, train_y
#
#
# # 获取测试集
# def get_test_data(test_begin=train_size):
#     data_test = data[test_begin:]
#     mean = np.mean(data_test, axis=0)
#     std = np.std(data_test, axis=0)
#     normalized_test_data = (data_test-mean)/std  # 标准化
#     size = (len(normalized_test_data)+time_step-1)//time_step  # 有size个sample
#     test_x, test_y = [], []
#     for i in range(size-1):
#         x = normalized_test_data[i*time_step:(i+1)*time_step, :11]
#         y = normalized_test_data[i*time_step:(i+1)*time_step, 11:14]
#         test_x.append(x.tolist())
#         test_y.extend(y.tolist())
#     return mean, std, test_x, test_y
#
#
# # ——————————————————定义神经网络变量——————————————————
# X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
# Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
#
# # 输入层、输出层权重、偏置
#
# W_in = tf.Variable(tf.random_normal([input_size, rnn_unit]))
# b_in = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
# W_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
# b_out = tf.Variable(tf.constant(0.1, shape=[3, ]))
#
#
# # ——————————————————定义神经网络变量——————————————————
# def lstm(X):
#     batch_size = tf.shape(X)[0]
#     time_step = tf.shape(X)[1]
#     input_data = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
#     input_rnn = tf.matmul(input_data, W_in)+b_in
#     input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
#     cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
#     init_state = cell.zero_state(batch_size, dtype=tf.float32)
#
#     # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
#     output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
#     output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
#     pred = tf.matmul(output, W_out)+b_out
#     return pred, final_states
#
#
# # ——————————————————训练模型——————————————————
# def train_lstm(batch_size=80, train_end=train_size):
#
#     batch_index, train_x, train_y = get_train_data(batch_size, train_end)
#
#     train_y = np.array(train_y)
#
#     # 损失函数
#
#
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         with graph.as_default():
#             pred, _ = lstm(X)
#             loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
#             train_op = tf.train.AdamOptimizer(lr).minimize(loss)
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             # 重复训练1000次
#             for i in range(100):
#                 total_loss = 0
#                 for step in range(len(batch_index)-1):
#                     _, loss_ = sess.run([train_op, loss],
#                                         feed_dict={X: train_x[batch_index[step]:batch_index[step+1]],
#                                                    Y: train_y[batch_index[step]:batch_index[step+1]]
#                                                    })
#                     total_loss += loss_
#                 if i % 100 == 0:
#                     print(i, total_loss)
#
#             saver.save(sess, 'lstm_model/test')
#
#
# # ————————————————预测模型————————————————————
# def verify():
#
#     mean, std, test_x, test_y = get_test_data(train_size)
#
#     graph1 = tf.Graph()
#     with tf.Session(graph=graph1) as sess:
#         with graph1.as_default():
#             pred, _ = lstm(X)
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             module_file = tf.train.latest_checkpoint('lstm_model/')
#             saver.restore(sess, module_file)
#             test_predict = []
#             for step in range(len(test_x)-1):
#                 prob = sess.run(pred, feed_dict={X: [test_x[step]]})
#                 test_predict.extend(prob)
#             test_y = np.array(test_y)*std[11:14]+mean[11:14]
#             test_predict = np.array(test_predict)*std[11:14]+mean[11:14]
#             acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  # 偏差
#             # 以折线图表示结果
#             plt.figure()
#             plt.subplot(2, 2, 1)
#             plt.plot(list(range(len(test_predict))), test_predict[:, 0], color='b', label="predict")
#             plt.legend(loc='upper right')
#             plt.subplot(2, 2, 2)
#             plt.plot(list(range(len(test_y))), test_y[:, 0],  color='r', label="real")
#             plt.legend(loc='upper right')
#             plt.subplot(2, 1, 2)
#
#             plt.plot(list(range(len(test_y))), test_y[:, 0], color='r', label="real")
#             plt.plot(list(range(len(test_predict))), test_predict, color='b', label="predict")
#
#             plt.legend(loc='upper right')
#             print(acc)
#             plt.show()
#             plt.savefig('lstm_test/test.png')
#             plt.close()
#
#
# # with tf.Session() as sess:
#     #     # 参数恢复
#     #     module_file = tf.train.latest_checkpoint('lstm_model/')
#     #     saver.restore(sess, module_file)
#     #     test_predict = []
#     #     for step in range(len(test_x)-1):
#     #         prob = sess.run(pred, feed_dict={X: [test_x[step]]})
#     #         test_predict.extend(prob)
#     #     test_y = np.array(test_y)*std[11:14]+mean[11:14]
#     #     test_predict = np.array(test_predict)*std[11:14]+mean[11:14]
#     #     acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  # 偏差
#     #     # 以折线图表示结果
#     #     plt.figure()
#     #     plt.subplot(2, 2, 1)
#     #     plt.plot(list(range(len(test_predict))), test_predict[:, 0], color='b', label="predict")
#     #     plt.legend(loc='upper right')
#     #     plt.subplot(2, 2, 2)
#     #     plt.plot(list(range(len(test_y))), test_y[:, 0],  color='r', label="real")
#     #     plt.legend(loc='upper right')
#     #     plt.subplot(2, 1, 2)
#     #
#     #     plt.plot(list(range(len(test_y))), test_y[:, 0], color='r', label="real")
#     #     plt.plot(list(range(len(test_predict))), test_predict, color='b', label="predict")
#     #
#     #     plt.legend(loc='upper right')
#     #     print(acc)
#     #     plt.show()
#     #     plt.savefig('lstm_test/test.png')
#     #     plt.close()
#
#
# def prediction(predict_x):
#
#     mean, std, test_x, test_y = get_test_data(train_size)
#     pred, _ = lstm(X)
#     predict_x = (predict_x - mean[0:11])/std[0:11]
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         # 参数恢复
#         module_file = tf.train.latest_checkpoint('lstm_model/')
#         saver.restore(sess, module_file)
#         prob = sess.run(pred, feed_dict={X: [predict_x]})
#
#         print(prob)
#         test_predict = np.array(prob) * std[11:14] + mean[11:14]
#         return test_predict
#
#
# if __name__ == "__main__":
#     predict_x = [[170.0, 168.0, 167.0, 151.0, 139.0, 174.0, 136.0, 120.0, 153.0, 61.0, 33.0, 119.0, 141.0, 148.0],
#                  [168.0, 167.0, 151.0, 139.0, 174.0, 119.0, 120.0, 153.0, 133.0, 61.0, 33.0, 141.0, 148.0, 107.0],
#                  [167.0, 151.0, 139.0, 174.0, 119.0, 141.0, 153.0, 133.0, 127.0, 61.0, 33.0, 148.0, 107.0, 146.0],
#                  [151.0, 139.0, 174.0, 119.0, 141.0, 148.0, 133.0, 127.0, 129.0, 61.0, 33.0, 107.0, 146.0, 130.0],
#                  [139.0, 174.0, 119.0, 141.0, 148.0, 107.0, 127.0, 129.0, 146.0, 61.0, 33.0, 146.0, 130.0, 116.0],
#                  [174.0, 119.0, 141.0, 148.0, 107.0, 146.0, 129.0, 146.0, 130.0, 61.0, 33.0, 130.0, 116.0, 126.0],
#                  [119.0, 141.0, 148.0, 107.0, 146.0, 130.0, 146.0, 130.0, 105.0, 61.0, 33.0, 116.0, 126.0, 115.0],
#                  [141.0, 148.0, 107.0, 146.0, 130.0, 116.0, 130.0, 105.0, 119.0, 61.0, 33.0, 126.0, 115.0, 93.0],
#                  [148.0, 107.0, 146.0, 130.0, 116.0, 126.0, 105.0, 119.0, 120.0, 61.0, 33.0, 115.0, 93.0, 119.0],
#                  [107.0, 146.0, 130.0, 116.0, 126.0, 115.0, 119.0, 120.0, 99.0, 61.0, 33.0, 93.0, 119.0, 103.0],
#                  [146.0, 130.0, 116.0, 126.0, 115.0, 93.0, 120.0, 99.0, 95.0, 61.0, 33.0, 119.0, 103.0, 106.0],
#                  [130.0, 116.0, 126.0, 115.0, 93.0, 119.0, 99.0, 95.0, 105.0, 61.0, 33.0, 103.0, 106.0, 104.0],
#                  [116.0, 126.0, 115.0, 93.0, 119.0, 103.0, 95.0, 105.0, 100.0, 61.0, 33.0, 106.0, 104.0, 99.0],
#                  [126.0, 115.0, 93.0, 119.0, 103.0, 106.0, 105.0, 100.0, 92.0, 61.0, 33.0, 104.0, 99.0, 107.0],
#                  [115.0, 93.0, 119.0, 103.0, 106.0, 104.0, 100.0, 92.0, 91.0, 61.0, 33.0, 99.0, 107.0, 94.0],
#                  [93.0, 119.0, 103.0, 106.0, 104.0, 99.0, 92.0, 91.0, 85.0, 61.0, 33.0, 107.0, 94.0, 88.0],
#                  [119.0, 103.0, 106.0, 104.0, 99.0, 107.0, 91.0, 85.0, 84.0, 61.0, 33.0, 94.0, 88.0, 81.0],
#                  [103.0, 106.0, 104.0, 99.0, 107.0, 94.0, 85.0, 84.0, 79.0, 61.0, 33.0, 88.0, 81.0, 117.0],
#                  [106.0, 104.0, 99.0, 107.0, 94.0, 88.0, 84.0, 79.0, 80.0, 61.0, 33.0, 81.0, 117.0, 94.0],
#                  [104.0, 99.0, 107.0, 94.0, 88.0, 81.0, 79.0, 80.0, 73.0, 11.0, 33.0, 117.0, 94.0, 71.0]]
#     predict_x = np.array(predict_x)
#     # print(prediction(predict_x[:, :11]))
#     train_lstm()
#     verify()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# 定义常量
# rnn_unit = 10  # hidden layer units
# input_size = 11
# time_step = 20
# output_size = 3
# lr = 0.0006  # 学习率
# # # ——————————————————导入数据——————————————————————
# # f = open('LSTM/sample/lstm_data.csv')
# # df = pd.read_csv(f)  # 读入数据
# # data = df.iloc[:, 1:15].values  # 取第1-12列
# # train_size = int(len(data) * 0.7)
#
#
# def load_data(filename='Integration/sample/bp_ga_data.csv'):
#     data = []
#     i = 1
#     with open(filename, 'r') as f:
#         for line in f:
#             sample = line.strip().split(',')
#             if len(sample) == 15 and i >= 577:
#                 data.append([float(sample[1]), float(sample[2]), float(sample[3]),
#                             float(sample[4]), float(sample[5]), float(sample[6]),
#                             float(sample[7]), float(sample[8]), float(sample[9]),
#                             float(sample[10]), float(sample[11]), float(sample[12]),
#                             float(sample[13]), float(sample[14])])
#             i += 1
#     data = np.array(data)
#     return data
#
#
# # 获取训练集
# def get_train_data(batch_size=60):
#     data = load_data()
#     batch_index = []
#     train_end = int(0.7*len(data))
#     data_train = data[:train_end]
#     normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
#     train_x, train_y = [], []  # 训练集
#     for i in range(len(normalized_train_data) - time_step):
#         if i % batch_size == 0:
#             batch_index.append(i)
#         x = normalized_train_data[i:i + time_step, 3:14]
#         y = normalized_train_data[i:i + time_step, 0:3]
#         train_x.append(x.tolist())
#         train_y.append(y.tolist())
#     batch_index.append((len(normalized_train_data) - time_step))
#     return batch_index, train_x, train_y
#
#
# # 获取测试集
# def get_test_data():
#     data = load_data()
#     test_begin = int(0.7*len(data))
#     data_test = data[test_begin:]
#     mean = np.mean(data_test, axis=0)
#     std = np.std(data_test, axis=0)
#     normalized_test_data = (data_test - mean) / std  # 标准化
#     size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
#     test_x, test_y = [], []
#     for i in range(size - 1):
#         x = normalized_test_data[i * time_step:(i + 1) * time_step, 3:14]
#         y = normalized_test_data[i * time_step:(i + 1) * time_step, 0:3]
#         test_x.append(x.tolist())
#         test_y.extend(y.tolist())
#     return mean, std, test_x, test_y
#
#
# # ——————————————————训练模型——————————————————
# def train_lstm(batch_size=80):
#     batch_index, train_x, train_y = get_train_data(batch_size)
#     train_y = np.array(train_y)
#
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         with graph.as_default():
#
#             # ——————————————————定义神经网络变量——————————————————
#             X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
#             Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
#             # 输入层、输出层权重、偏置
#
#             W_in = tf.Variable(tf.random_normal([input_size, rnn_unit]))
#             b_in = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
#             W_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
#             b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
#             batch_size = tf.shape(X)[0]
#             input_data = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
#             input_rnn = tf.matmul(input_data, W_in) + b_in
#             input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
#             cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
#             init_state = cell.zero_state(batch_size, dtype=tf.float32)
#
#             # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
#             output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
#             output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
#             pred = tf.matmul(output, W_out) + b_out
#             loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
#             train_op = tf.train.AdamOptimizer(lr).minimize(loss)
#
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             # 重复训练1000次
#             for i in range(10):
#                 total_loss = 0
#                 for step in range(len(batch_index) - 1):
#                     _, loss_ = sess.run([train_op, loss],
#                                         feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
#                                                    Y: train_y[batch_index[step]:batch_index[step + 1]]
#                                                    })
#                     total_loss += loss_
#                 print(i, total_loss)
#
#             saver.save(sess, 'lstm_model/test')
#
#
# # ————————————————预测模型————————————————————
# def verify():
#     mean, std, test_x, test_y = get_test_data()
#
#     graph1 = tf.Graph()
#     with tf.Session(graph=graph1) as sess:
#         with graph1.as_default():
#             # ——————————————————定义神经网络变量——————————————————
#             X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
#             Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
#             # 输入层、输出层权重、偏置
#
#             W_in = tf.Variable(tf.random_normal([input_size, rnn_unit]))
#             b_in = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
#             W_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
#             b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
#             batch_size = tf.shape(X)[0]
#             input_data = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
#             input_rnn = tf.matmul(input_data, W_in) + b_in
#             input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
#             cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
#             init_state = cell.zero_state(batch_size, dtype=tf.float32)
#
#             # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
#             output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
#             output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
#             pred = tf.matmul(output, W_out) + b_out
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             module_file = tf.train.latest_checkpoint('lstm_model/')
#             saver.restore(sess, module_file)
#             test_predict = []
#             for step in range(len(test_x) - 1):
#                 prob = sess.run(pred, feed_dict={X: [test_x[step]]})
#                 test_predict.extend(prob)
#             test_y = np.array(test_y) * std[11:14] + mean[11:14]
#             test_predict = np.array(test_predict) * std[11:14] + mean[11:14]
#             acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
#             # 以折线图表示结果
#             plt.figure()
#             plt.subplot(2, 2, 1)
#             plt.plot(list(range(len(test_predict))), test_predict[:, 0], color='b', label="predict")
#             plt.legend(loc='upper right')
#             plt.subplot(2, 2, 2)
#             plt.plot(list(range(len(test_y))), test_y[:, 0], color='r', label="real")
#             plt.legend(loc='upper right')
#             plt.subplot(2, 1, 2)
#
#             plt.plot(list(range(len(test_y))), test_y[:, 0], color='r', label="real")
#             plt.plot(list(range(len(test_predict))), test_predict, color='b', label="predict")
#
#             plt.legend(loc='upper right')
#             print(acc)
#             plt.show()
#             plt.savefig('test.png')
#             plt.close()
#
#
# def prediction(predict_x):
#     mean, std, test_x, test_y = get_test_data()
#
#     graph2 = tf.Graph()
#     with tf.Session(graph=graph2) as sess:
#         with graph2.as_default():
#             # ——————————————————定义神经网络变量——————————————————
#             X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
#             Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
#             # 输入层、输出层权重、偏置
#
#             W_in = tf.Variable(tf.random_normal([input_size, rnn_unit]))
#             b_in = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
#             W_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
#             b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
#             batch_size = tf.shape(X)[0]
#             input_data = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
#             input_rnn = tf.matmul(input_data, W_in) + b_in
#             input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
#             cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
#             init_state = cell.zero_state(batch_size, dtype=tf.float32)
#
#             # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
#             output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
#             output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
#             pred = tf.matmul(output, W_out) + b_out
#             predict_x = (predict_x - mean[0:11]) / std[0:11]
#             # 参数恢复
#             saver = tf.train.Saver()
#             module_file = tf.train.latest_checkpoint('lstm_model/')
#             saver.restore(sess, module_file)
#             prob = sess.run(pred, feed_dict={X: [predict_x]})
#
#             print(prob)
#             test_predict = np.array(prob) * std[11:14] + mean[11:14]
#             return test_predict


class My_LSTM:
    def __init__(self, rnn_unit, input_size, time_step, output_size, lr, data):
        self.rnn_unit = rnn_unit
        self.input_size = input_size
        self.time_step = time_step
        self.output_size = output_size
        self.lr = lr
        self.data = data

    # ———————————————获取训练集——————————————————
    def get_train_data(self, batch_size=60):
        batch_index = []
        train_size = int(0.7 * len(self.data))
        data_train = self.data[:train_size]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data) - self.time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + self.time_step, 3:14]
            y = normalized_train_data[i:i + self.time_step, 0:3]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - self.time_step))
        return batch_index, train_x, train_y

    # ———————————————获取验证集——————————————————
    def get_test_data(self):
        test_begin = int(0.7 * len(self.data))
        data_test = self.data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std  # 标准化
        size = (len(normalized_test_data) + self.time_step - 1) // self.time_step  # 有size个sample
        test_x, test_y = [], []
        for i in range(size - 1):
            x = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, 3:14]
            y = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, 0:3]
            test_x.append(x.tolist())
            test_y.extend(y.tolist())
        return mean, std, test_x, test_y

    # ———————————————训练模型——————————————————
    def train_lstm(self, batch_size=80):
        batch_index, train_x, train_y = self.get_train_data(batch_size)
        train_y = np.array(train_y)

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():

                # ——————————————————定义神经网络变量——————————————————
                X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
                Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])
                # 输入层、输出层权重、偏置

                w_in = tf.Variable(tf.random_normal([self.input_size, self.rnn_unit]))
                b_in = tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ]))
                w_out = tf.Variable(tf.random_normal([self.rnn_unit, self.output_size]))
                b_out = tf.Variable(tf.constant(0.1, shape=[self.output_size, ]))
                batch_size = tf.shape(X)[0]
                time_step = tf.shape(X)[1]
                input_data = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
                input_rnn = tf.matmul(input_data, w_in) + b_in
                input_rnn = tf.reshape(input_rnn, [-1, time_step, self.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
                init_state = cell.zero_state(batch_size, dtype=tf.float32)

                # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
                output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
                output = tf.reshape(output_rnn, [-1, self.rnn_unit])  # 作为输出层的输入
                pred = tf.matmul(output, w_out) + b_out
                loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
                train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                # 重复训练1000次
                train_loss = []
                for i in range(1000):
                    total_loss = 0
                    for step in range(len(batch_index) - 1):
                        _, loss_ = sess.run([train_op, loss],
                                            feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                       Y: train_y[batch_index[step]:batch_index[step + 1]]
                                                       })
                        total_loss += loss_
                    train_loss.append(total_loss)
                saver.save(sess, 'lstm_model/test')
                plt.figure()
                plt.plot(train_loss, 'r.', label='loss')
                plt.xlabel("Train time")
                plt.ylabel("Mean Squared Error")
                plt.title('Learning curves')
                plt.legend(loc="best")
                plt.savefig("lstm_loss/loss.png")
                plt.close()

    # ———————————————预测模型——————————————————
    def verify(self):
        mean, std, test_x, test_y = self.get_test_data()

        graph1 = tf.Graph()
        with tf.Session(graph=graph1) as sess:
            with graph1.as_default():
                # ——————————————————定义神经网络变量——————————————————
                X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
                Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])
                # 输入层、输出层权重、偏置

                w_in = tf.Variable(tf.random_normal([self.input_size, self.rnn_unit]))
                b_in = tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ]))
                w_out = tf.Variable(tf.random_normal([self.rnn_unit, self.output_size]))
                b_out = tf.Variable(tf.constant(0.1, shape=[self.output_size, ]))
                batch_size = tf.shape(X)[0]
                input_data = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
                input_rnn = tf.matmul(input_data, w_in) + b_in
                input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
                init_state = cell.zero_state(batch_size, dtype=tf.float32)

                # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
                output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
                output = tf.reshape(output_rnn, [-1, self.rnn_unit])  # 作为输出层的输入
                pred = tf.matmul(output, w_out) + b_out
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                module_file = tf.train.latest_checkpoint('lstm_model/')
                saver.restore(sess, module_file)
                test_predict = []
                for step in range(len(test_x) - 1):
                    prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                    test_predict.extend(prob)
                test_y = np.array(test_y) * std[11:14] + mean[11:14]
                test_predict = np.array(test_predict) * std[11:14] + mean[11:14]

                acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
                model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error,
                                      r2_score]  # 回归评估指标对象集
                model_metrics_list = []  # 回归评估指标列表

                tmp_list = []  # 每个内循环的临时结果列表
                for m in model_metrics_name:  # 循环每个指标对象
                    tmp_score = m(test_y[:len(test_predict), 0], test_predict[:, 0])  # 计算每个回归指标结果
                    tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
                tmp_list.append(acc)
                model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

                df2 = pd.DataFrame(model_metrics_list, index=['lstm'],
                                   columns=['ev', 'mae', 'mse', 'r2', 'acc'])  # 建立回归指标的数据框

                print('regression metrics:')  # 打印输出标题
                print(df2)  # 打印输出回归指标的数据框
                print(70 * '-')  # 打印分隔线
                print('short name \t full name')  # 打印输出缩写和全名标题
                print(' ev \t\t\t explained_variance')
                print('mae \t\t\t mean_absolute_error')
                print('mse \t\t\t mean_squared_error')
                print(' r2 \t\t\t r2')
                print('acc \t\t\t 相对误差')
                print(70 * '-')  # 打印分隔线

                epoch = int(len(test_predict) / 288)
                for i in range(epoch):
                    plt.figure()
                    plt.xlabel("time")
                    plt.ylabel("number")
                    plt.plot(test_predict[i * 288: (i + 1) * 288, 0:1], 'r', label="Predictive value")
                    plt.plot(test_y[i * 288: (i + 1) * 288, 0:1], 'b', label="Actual value")
                    plt.legend(loc='upper right')
                    plt.savefig('lstm_test/test%d.png' % (i + 1))
                    plt.close()

    def prediction(self, predict_x):
        mean, std, test_x, test_y = self.get_test_data()
        predict_x = (predict_x - mean[3:14]) / std[3:14]
        graph2 = tf.Graph()
        with tf.Session(graph=graph2) as sess:
            with graph2.as_default():
                # ——————————————————定义神经网络变量——————————————————
                X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
                Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])
                # 输入层、输出层权重、偏置

                w_in = tf.Variable(tf.random_normal([self.input_size, self.rnn_unit]))
                b_in = tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ]))
                w_out = tf.Variable(tf.random_normal([self.rnn_unit, self.output_size]))
                b_out = tf.Variable(tf.constant(0.1, shape=[self.output_size, ]))
                batch_size = tf.shape(X)[0]
                input_data = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
                input_rnn = tf.matmul(input_data, w_in) + b_in
                input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
                init_state = cell.zero_state(batch_size, dtype=tf.float32)

                # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
                output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                             dtype=tf.float32)
                output = tf.reshape(output_rnn, [-1, self.rnn_unit])  # 作为输出层的输入
                pre_y = tf.matmul(output, w_out) + b_out

                sess.run(tf.global_variables_initializer())

                # 参数恢复
                saver = tf.train.Saver()
                module_file = tf.train.latest_checkpoint('lstm_model/')
                saver.restore(sess, module_file)
                prob = sess.run(pre_y, feed_dict={X: [predict_x]})

                test_predict = np.array(prob) * std[0:3] + mean[0:3]
                return test_predict[-1]
