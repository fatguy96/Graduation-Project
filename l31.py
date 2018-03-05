import tensorflow as tf


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
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    col1_batch, col2_batch, col3_batch = tf.train.shuffle_batch(
        [col1, col2, col3], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
    return col1_batch, col2_batch, col3_batch


file_names = ['test.csv']
batch_size = 24920
num_epochs = 1

a1, a2, a3 = input_pipeline(file_names, batch_size, num_epochs)

with tf.Session() as sess:

    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            batchs = sess.run([a1, a2, a3])
            print(batchs)
    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')
    finally:
        coord.request_stop()

    coord.join(threads)
