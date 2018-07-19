#from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    #x = [5x10000 (5 rows)]
    y = y.reshape((batch_size, -1))
    #y = [5x10000 (5 rows)]

    # last 3 elements in the row and predict at the beginning of the next row
    # print(x[:,0:3])
    # print(y[:,3:6])
    # print(x[:,-3:])
    # print(y[[1,2,3,4,0],0:3])

    return (x, y)
# 5 x 15
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# 5 x 4
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# 5 x 4
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
# 1 x 4
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

# 4 x 2
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
# 1 x 2
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

print("Input X: ", batchX_placeholder.get_shape())

# unstack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
# list: 1st input batch [1..5], 2nd input batch [1..5] )

print("Input series: ", len(inputs_series))

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    # vector of 1 input (but it is column of 5) (1x5)
    current_input = tf.reshape(current_input, [batch_size, 1])
    # input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # Increasing number of columns
    # input = 5x1 ,, state=5x4
    input_and_state_concatenated = tf.concat(axis=1, values=[current_input, current_state])  # Increasing number of columns
    # (5x5)

    # 5x5 xx W == 5 x 4 => 5x4
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state

# state=5x4 xx 4x2 => 5x2
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()