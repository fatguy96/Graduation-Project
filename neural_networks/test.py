import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = np.loadtxt("../fire_theft.csv", delimiter=",")
train_set_x_orig = np.array(dataset[:, 0])  # your train set features
train_set_y_orig = np.array(dataset[:, 1])  # your train set labels
x = train_set_x_orig.reshape(-1, 1)
y = train_set_y_orig.reshape(-1, 1)
print(x[1].reshape(-1, 1).shape)


X = tf.placeholder(tf.float32, shape=(None, 1), name='X')
Y = tf.placeholder(tf.float32, shape=(None, 1), name='Y')

W = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

Y_predicted = X * W + b

loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
n_samples = train_set_x_orig.shape[0]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(50):
        total_loss = 0
        for j in range(x.shape[0]):
            _, l = sess.run([optimizer, loss], feed_dict={X: x[j].reshape(-1, 1), Y: y[j].reshape(-1, 1)})
            total_loss += l
        plt.plot(i, total_loss/n_samples, 'r-o')

        print('Epoch{0}:{1}'.format(i, total_loss/n_samples))
    plt.show()
    w_value, b_value = sess.run([W, b])

print(w_value.shape)
plt.plot(train_set_x_orig, train_set_y_orig, 'bo', label='Real data')
plt.plot(train_set_x_orig, train_set_x_orig * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()


# print(X)c
# print(Y)
# #Z = dataset[:, 4]
# print(dataset.shape)
# print(X.size)
# print(Y.size)
# standardize_X = preprocessing.scale(X.reshape(-1, 1))
# normalized_X = preprocessing.normalize(X.reshape(-1, 1))
# normalized_Y = preprocessing.scale(Y.reshape(-1, 1))
# print(standardize_X.shape)
# print(normalized_X.shape)
# print(normalized_Y.shape)
#
# print(normalized_X[1,1])
# print(standardize_X)
# print(normalized_X)
# print(dataset.shape)
# print(dataset.size)
# plt.figure(1)
# plt.title("sum of car")
# ax1 = plt.subplot(2, 2, 1)
# ax2 = plt.subplot(2, 2, 2)
# ax3 = plt.subplot(2, 2, 3)
# ax4 = plt.subplot(2, 2, 4)
# for i in range(288):
#     plt.plot(i, X[i], 'r-+')
#     plt.sca(ax1)
#     ax1.set_title("first day")
#     plt.plot(i+288, X[i+288], 'r-+')
#     plt.sca(ax2)
#     ax2.set_title("second day")
#     plt.plot(i + 288*2, X[i + 288*2], 'r-+')
#     plt.sca(ax3)
#     ax3.set_title("three")
#     plt.plot(i + 288*3, X[i + 288*3], 'r-+')
#     plt.sca(ax4)
#     ax4.set_title("four")
# plt.show()