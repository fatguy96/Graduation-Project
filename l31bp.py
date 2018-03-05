import tensorflow as tf
import matplotlib.pyplot as plt


# TODO：根据后期的确定的样本进行修改 现在的样本为前两天当时的数据来确定今天当时的数据
def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.0], [0.0], [0.0]]
    col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
    return col1, col2, col3


def input_pipeline(filenames, batch_size, num_epochs=1):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=False)
    col1, col2, col3 = read_my_file_format(filename_queue)
    example = tf.stack([col2, col3], 0)
    label = tf.stack([col1], 0)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
    return example_batch, label_batch


file_names = ['test.csv']
batch_size = 64
batch_size1 = 24920

num_epochs = 50

example_batch, label_batch = input_pipeline(file_names, batch_size1, num_epochs)

W1 = tf.Variable(tf.random_normal([2, 4], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([1, 4], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([4, 1], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

# TODO: 1、优化神经网络，模型就是一个单层神经网络
# TODO：2、考虑如何算出总的代价，由于当前的样本数目为25920，选择使用小batch进行训练，训练后如何查看其在全局的性能
# TODO：3、对样本进行正则化， 如何对全局数据，现在只是对单个batch进行处理
z = tf.add(tf.matmul(x, W1), b1)
a = tf.nn.relu(z, "a")
y = tf.add(tf.matmul(a, W2), b2)

# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
loss = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # for i in range(10):
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    total_loss = 0
    i = 1
    try:
        while not coord.should_stop():

            example_batchs, label_batchs = sess.run([example_batch, label_batch])

            [label_batch_mean, label_batch_varia] = tf.nn.moments(label_batch, axes=[0])

            [example_batchs_mean, example_batchs_varia] = tf.nn.moments(example_batch, axes=[0])

            offset = 0

            scale = 0.1

            vari_epsl = 0.0001

            # 正则化
            example_batchs_bn = tf.nn.batch_normalization(example_batchs, example_batchs_mean, example_batchs_varia,
                                                          offset, scale, vari_epsl)
            label_batchs_bn = tf.nn.batch_normalization(label_batchs, label_batch_mean, label_batch_varia,
                                                       offset, scale, vari_epsl)
            example_test = sess.run(example_batchs_bn)

            label_test = sess.run(label_batchs_bn)

            sess.run(train_step, feed_dict={x: example_test, y_: label_test})

            loss_t = sess.run(loss, feed_dict={x: example_test, y_: label_test})

            print('epochs{}:{}'.format(i, loss_t))

            plt.plot(i, loss_t, 'r-o')

            i = i + 1

    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(threads)
    plt.show()
    print("W1:", W1.eval(), "b1:", b1.eval())
    print("W2:", W2.eval(), "b2:", b2.eval())
