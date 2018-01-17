print("start")

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros((784, 10),dtype=tf.float32))
b = tf.Variable(tf.zeros(10))

y_real = tf.placeholder(tf.float32, [None, 10])

#y_pred = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W)+b,y_real)

y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_real * tf.log(y_pred), reduction_indices=[1]))
# step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
# ==> this produces accuracy: 0.919

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y_pred))
step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# ==> this produces accuracy: 0.9055


correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ses = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    ses.run(step, feed_dict={x: batch_xs, y_real: batch_ys})
    print(ses.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))
