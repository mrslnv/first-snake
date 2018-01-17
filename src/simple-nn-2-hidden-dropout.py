import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(133)
sample_x, sample_y = ds.make_moons(1000,noise=.04)

plt.scatter(sample_x[:,0],sample_x[:,1],c=sample_y)
# plt.show()

# tensor
sess = tf.InteractiveSession()

# Simplest: [2x2] - linear regression x1,x2 input -> 0,1 output
initializer = tf.contrib.layers.xavier_initializer()
# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W = tf.get_variable("W",shape=[2,6],initializer=initializer)
b = tf.get_variable("b",shape=[6],initializer=initializer)

# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W1 = tf.get_variable("W1",shape=[6,6],initializer=initializer)
b1 = tf.get_variable("b1",shape=[6],initializer=initializer)

# W = tf.Variable(initializer, dtype=tf.float32, expected_shape=[2, 2])
W2 = tf.get_variable("W2",shape=[6,2],initializer=initializer)
b2 = tf.get_variable("b2",shape=[2],initializer=initializer)

x = tf.placeholder(tf.float32,[None, 2],name="x")

# NN
logits = tf.matmul(x,W) + b

relu = tf.nn.tanh(logits)
# relu = tf.nn.relu(logits)

logits1 = tf.matmul(relu, W1) + b1

# relu2 = tf.nn.relu(logits1)
relu2 = tf.nn.tanh(logits1)

dropout = tf.nn.dropout(relu2,0.7)

logits2 = tf.matmul(dropout, W2) + b2

y_ = tf.placeholder(tf.int32, [None],name="y_")

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2,y_,name="loss")

glob_init = tf.global_variables_initializer()
sess.run(glob_init)

optimizer = tf.train.GradientDescentOptimizer(0.00067)
train = optimizer.minimize(loss)

for _ in range(5000):
    sess.run(train, feed_dict={x: sample_x, y_: sample_y})

logits_argmax = tf.argmax(logits2,1,name="logits_argmax")

correct_prediction = tf.equal(tf.to_int32(logits_argmax), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: sample_x, y_: sample_y}))

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

plot_decision_boundary(lambda x: predict(x), sample_x,sample_y)
