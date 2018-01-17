import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sample(ratio, *arrays):
    ret = []
    a = arrays[0]
    l = len(a)
    sampler = np.random.choice(np.arange(l), int(round(l * ratio)), replace=False)
    for a in arrays:
        ret.append(a[sampler])
    return ret

class SimpleNN:
    def __init__(self):
        self.initializer = tf.contrib.layers.xavier_initializer()
    def build_layer(self, input_x,input_size, output_size, scope,drop_out):
        # ToDo: input_size from input_x
        with tf.name_scope(scope):
            with tf.variable_scope(scope):
                W = tf.get_variable("W", dtype=tf.float32,shape=[input_size, output_size], initializer=self.initializer)
                # ToDo: initialize with zeros
                b = tf.get_variable("b", dtype=tf.float32, shape=[output_size], initializer=self.initializer)

                logits = tf.matmul(input_x, W) + b

                activation = tf.nn.tanh(logits)
                dropout = tf.nn.dropout(activation, keep_prob=float(drop_out),name="drop")
                return dropout




np.random.seed(133)
all_x, all_y = ds.make_moons(2000, noise=.1)
sample_x, sample_y = sample(0.5, all_x, all_y)

plt.scatter(sample_x[:, 0], sample_x[:, 1], c=sample_y)
# plt.show()

# tensor
sess = tf.InteractiveSession()

nn = SimpleNN()

x = tf.placeholder(tf.float32, [None, 2], name="x")

l1 = nn.build_layer(x,2,8,"input-L",1)

l2 = nn.build_layer(l1,8,8,"hidden-L1",0.95)

l3 = nn.build_layer(l2,8,6,"hidden-L2",0.95)

l4 = nn.build_layer(l3,6,2,"output-L",0.95)

with tf.name_scope("loss"):
    y_ = tf.placeholder(tf.int32, [None], name="y_")

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(l4, y_, name="loss")
    reduce = tf.reduce_mean(loss)
    tf.summary.scalar("loss-reduced",reduce)

# tf.summary.scalar('Loss', loss)
# tf.summary.tensor_summary('Loss', loss)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train',
                                     sess.graph)

glob_init = tf.global_variables_initializer()
sess.run(glob_init)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(0.00067)
    train = optimizer.minimize(loss)

with tf.name_scope("accu"):
    logits_argmax = tf.argmax(l4, 1, name="logits_argmax")
    correct_prediction = tf.equal(tf.to_int32(logits_argmax), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

for i in range(5000):
    sess.run(train, feed_dict={x: sample_x, y_: sample_y})
    # summary, _ = sess.run([merged,train], feed_dict={x: sample_x, y_: sample_y})
    # train_writer.add_summary(summary, i)


print(sess.run(accuracy, feed_dict={x: all_x, y_: all_y}))

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

sess.run(merged, feed_dict={x: all_x, y_: all_y})

plot_decision_boundary(lambda x: predict(x), sample_x, sample_y)
