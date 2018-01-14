import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import RandomState

batch_size = 8

W1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_-input')

a = tf.matmul(x, W1)
y = tf.matmul(a, W2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001)  .minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

with tf.Session() as session:
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    print(session.run(W1))
    print(session.run(W2))

    Step = 5000
    for i in range(Step):
        start = (i*batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        session.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        total_cross_entropy = session.run(cross_entropy, feed_dict={x: X, y_: Y})
        print("after %d training step(s), cross_entropy on all data is %g" % (i, total_cross_entropy))
        plt.plot(i, total_cross_entropy, 'r-o')
    plt.show()
    print(session.run(W1))
    print(session.run(W2))

