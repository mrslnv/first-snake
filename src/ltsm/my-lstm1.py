import sklearn.datasets as ds
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np


x = [[i**2 for i in range(100)] for j in range(100)]

sess = tf.InteractiveSession()

X = tf.placeholder("x", [100, 100],dtype=tf.float32)

# tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0)
lstm_cell = rnn.BasicLSTMCell(10, forget_bias=1.0)

list = tf.unstack(x, 100, 1)

output, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


loss_op = tf.subtract(output,)
    logits=logits, labels=Y))
