1  # from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ToDo: experiment with - adding back batch
# ToDo: experiment with - without matrix W

# ToDo: experiment with more steps
# ToDo: experiment with more 2d input
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 8
num_batches = total_series_length // batch_size // truncated_backprop_length


def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows

    # x = [1x50000 (1 row)]

    y = y.reshape((batch_size, -1))

    # y = [1x50000 (1 rows)]

    return (x, y)


# 5 x 15
X_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name="x")
Y_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name="y")
init_state = (
    tf.placeholder(tf.float32, [batch_size, state_size], name="state_c"),
    tf.placeholder(tf.float32, [batch_size, state_size], name="state_h")
)

# 4 x 2
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32, name="W")
# 1 x 2
b2 = tf.Variable(np.zeros((num_classes)), dtype=tf.float32, name="b")

# unstack columns
# inputs_series = tf.unstack(X_placeholder, axis=1)
inputs_series = tf.split(value=X_placeholder, num_or_size_splits=truncated_backprop_length, axis=1)
labels_series = tf.unstack(Y_placeholder, axis=1)
# list: 1st input batch [1..5], 2nd input batch [1..5] )

cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
# states_series, current_state = tf.nn.static_rnn(cell=cell,inputs=inputs_series,dtype=tf.float32)
states_series, current_state = tf.nn.static_rnn(cell=cell, inputs=inputs_series, initial_state=init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]  # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
          zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    one_hot_output_series = np.array(predictions_series)[:, 0, :]
    single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

    plt.subplot(2, 3, 2)
    plt.cla()
    plt.axis([0, truncated_backprop_length, 0, 2])
    left_offset = range(truncated_backprop_length)
    plt.bar(left_offset, batchX[0, :], width=1, color="blue")
    plt.bar(left_offset, batchY[0, :] * 0.5, width=1, color="red")
    plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    writer = tf.summary.FileWriter('summary-log', sess.graph)

    for epoch_idx in range(num_epochs):
        x, y = generateData()
        # _current_state = (np.zeros((batch_size, state_size)), np.zeros((batch_size, state_size)))
        _current_state = (np.random.rand(batch_size, state_size), np.random.rand(batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    X_placeholder: batchX,
                    Y_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
