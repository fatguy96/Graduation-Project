import tensorflow as tf
import matplotlib.pyplot as plt


def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14 = \
        tf.decode_csv(value, record_defaults=record_defaults)
    return col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14


def input_pipeline(filename, batch_size, num_epochs=1):

    filename_queue = tf.train.string_input_producer(filename,
                                                    num_epochs=num_epochs,
                                                    shuffle=True)
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14\
        = read_my_file_format(filename_queue)

    example = tf.stack([col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14], 0)
    label = tf.stack([col1], 0)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True,
        num_threads=8
    )
    return example_batch, label_batch


W1 = tf.Variable(tf.random_normal([11, 9], stddev=1, seed=1, mean=0))
b1 = tf.Variable(tf.random_normal([9], stddev=1, seed=1, mean=0))
W2 = tf.Variable(tf.random_normal([9, 3], stddev=1, seed=1, mean=0))
b2 = tf.Variable(tf.random_normal([3], stddev=1, seed=1, mean=0))
W3 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1, mean=0))
b3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1, mean=0))

x = tf.placeholder(tf.float32, shape=(None, 11), name='example')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1, "a1")
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.relu(z2)
y = tf.add(tf.matmul(a2, W3), b3)

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.square(y_ - y))
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.9, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

file_names = ['../sample/bp_train.csv']
batch_size = 32
num_epochs = 10
example_batch, label_batch = input_pipeline(file_names, batch_size, num_epochs)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 1
    plt.title('lose of each batch')
    plt.xlabel('batch')
    plt.ylabel('lose')
    try:
        while not coord.should_stop():
            example_batchs, label_batchs = sess.run([example_batch, label_batch])
            _, loss_ = sess.run([train_step, loss], feed_dict={x: example_batchs, y_: label_batchs})
            print('{} loss :{}'.format(step, loss_))
            plt.plot(step, loss_, 'b.')
            step = step+1
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
    saver.save(sess, 'my_model/test')
    plt.show()
