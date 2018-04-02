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
        filenames, num_epochs=num_epochs, shuffle=True)
    col1, col2, col3 = read_my_file_format(filename_queue)
    example = tf.stack([col2, col3], 0)
    label = tf.stack([col1], 0)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True,num_threads=8
    )
    return example_batch, label_batch


file_names = ['epoch_test.csv']
batch_size = 64
num_epochs = 50
example_batch, label_batch = input_pipeline(file_names, batch_size, num_epochs)
W1 = tf.Variable(tf.random_normal([2, 4], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([4], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([4, 1], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')
z = tf.add(tf.matmul(x, W1), b1)
a = tf.nn.relu(z, "a")
y = tf.add(tf.matmul(a, W2), b2)
loss = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    j = 1
    try:
        while not coord.should_stop():
            total_batch = int(25920/batch_size)
            total_loss = 0
            for i in range(total_batch):
                example_batchs, label_batchs = sess.run([example_batch, label_batch])

                # 正则化数据
                [label_batch_mean, label_batch_varia] = tf.nn.moments(label_batch, axes=[0])
                [example_batchs_mean, example_batchs_varia] = tf.nn.moments(example_batch, axes=[0])
                offset = 0
                scale = 0.1
                vari_epsl = 0.0001
                # 正则化
                # example_batchs_bn = tf.nn.batch_normalization(example_batchs, example_batchs_mean, example_batchs_varia,
                #                                            offset, scale, vari_epsl)
                # label_batchs_bn = tf.nn.batch_normalization(label_batchs, label_batch_mean, label_batch_varia,
                #                                            offset, scale, vari_epsl)
                # example_test, label_test = sess.run([example_batchs_bn, label_batchs_bn])
                _, loss_ = sess.run([train_step, loss], feed_dict={x: example_batchs, y_: label_batchs})
                total_loss += loss_
                print('batch {}: loss:{}'.format(i, loss_))
            print('epochs {}  :{}'.format(j, total_loss))
            plt.plot(j, total_loss, 'r-o')
            j = j + 1

    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
    plt.show()
    print("W1:", W1.eval(), "b1:", b1.eval())
    print("W2:", W2.eval(), "b2:", b2.eval())
