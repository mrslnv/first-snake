import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# this is super simple experiment with tensor board



np.random.seed(133)

# tensor
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, None, name="x")
tf.summary.scalar("dummy_1", x)

y = tf.placeholder(tf.float32, None, name="y")
tf.summary.scalar("dummy_2", y)

dist1 = tf.placeholder(tf.float32, [1000], name="dist1")
tf.summary.histogram("dist_1", dist1)

glob_init = tf.global_variables_initializer()
sess.run(glob_init)

summaryMerged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary-log', sess.graph)

for i in range(10):
    xx = 2*i
    yy = i**2
    histN = np.random.normal(2 * yy, xx, 1000)
    xx, yy, dd, sumOut = sess.run([x, y, dist1, summaryMerged], feed_dict={
        x: xx,
        y: yy,
        dist1: histN
    })
    print("x=", xx)
    print("y=", yy)
    print("mean=", np.mean(histN))
    writer.add_summary(sumOut, i)

writer.flush()
