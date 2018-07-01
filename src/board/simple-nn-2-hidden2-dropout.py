import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#this is simple-nn-2-hidden2-dropout + trying to add tensor board
#tensor board OK - histograms for every input

def sample(ratio, *arrays):
    ret = []
    a = arrays[0]
    l = len(a)
    sampler = np.random.choice(np.arange(l), int(round(l * ratio)), replace=False)
    for a in arrays:
        ret.append(a[sampler])
    return ret


np.random.seed(133)
all_x, all_y = ds.make_moons(2000, noise=.1)
sample_x, sample_y = sample(0.5, all_x, all_y)

plt.scatter(sample_x[:, 0], sample_x[:, 1], c=sample_y)
# plt.show()

# tensor
sess = tf.InteractiveSession()


# Simplest: [2x2] - linear regression x1,x2 input -> 0,1 output
initializer = tf.contrib.layers.xavier_initializer()
# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W = tf.get_variable("W", shape=[2, 8], initializer=initializer)
b = tf.get_variable("b", shape=[8], initializer=initializer)

tf.summary.histogram("weight_1",W)

# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W1 = tf.get_variable("W1", shape=[8, 8], initializer=initializer)
b1 = tf.get_variable("b1", shape=[8], initializer=initializer)

tf.summary.histogram("weight_2",W1)

# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W2 = tf.get_variable("W2", shape=[8, 6], initializer=initializer)
b2 = tf.get_variable("b2", shape=[6], initializer=initializer)

tf.summary.histogram("weight_3",W2)

# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W3 = tf.get_variable("W3", shape=[6, 2], initializer=initializer)
b3 = tf.get_variable("b3", shape=[2], initializer=initializer)

tf.summary.histogram("weight_4",W3)

x = tf.placeholder(tf.float32, [None, 2], name="x")

# NN
logits = tf.matmul(x, W) + b

tf.summary.histogram("logits_1",logits)

relu = tf.nn.tanh(logits)
# relu = tf.nn.relu(logits)

tf.summary.histogram("relu_1",relu)

logits1 = tf.matmul(relu, W1) + b1

tf.summary.histogram("logits_2",logits1)

# relu2 = tf.nn.relu(logits1)
relu2 = tf.nn.tanh(logits1)

tf.summary.histogram("relu_2",relu2)

dropout = tf.nn.dropout(relu2, 0.95)

tf.summary.histogram("dropout_1",dropout)

logits2 = tf.matmul(dropout, W2) + b2

tf.summary.histogram("logits_3",logits2)

# relu3 = tf.nn.relu(logits2)
relu3 = tf.nn.tanh(logits2)

tf.summary.histogram("relu_3",relu3)

dropout2 = tf.nn.dropout(relu3, 0.95)

tf.summary.histogram("dropout_2",dropout2)

logits3 = tf.matmul(dropout2, W3) + b3

tf.summary.histogram("logits_4",logits3)

y_ = tf.placeholder(tf.int32, [None], name="y_")

tf.summary.histogram("output",y_)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=y_, name="loss")

tf.summary.histogram("loss",loss)

glob_init = tf.global_variables_initializer()
sess.run(glob_init)

optimizer = tf.train.GradientDescentOptimizer(0.00067)
# really bad - oscillation
#optimizer = tf.train.GradientDescentOptimizer(0.01167)
train = optimizer.minimize(loss)
summaryMerged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary-log', sess.graph)

for i in range(5000):
    _, sumOut = sess.run([train, summaryMerged], feed_dict={x: sample_x, y_: sample_y})
    writer.add_summary(sumOut,i)

logits_argmax = tf.argmax(logits3, 1, name="logits_argmax")

correct_prediction = tf.equal(tf.to_int32(logits_argmax), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: all_x, y_: all_y}))

writer.flush()

def predict(x_in):
    print(x_in.shape)
    return sess.run(logits_argmax, feed_dict={x: x_in})


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    ravel_ = np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(ravel_)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


# print(predict([[0,0]]))

plot_decision_boundary(lambda x: predict(x), sample_x, sample_y)
