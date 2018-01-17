import tensorflow as tf
import copy
import numpy as np


def fce(x):
    return 3*x+8

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x_in = [i for i in range(8)]
y_out = [fce(xi) for xi in x_in]

io = tf.contrib.learn.io
input_fn = io.numpy_input_fn({"x":x_in}, y_out, batch_size=8,
                             num_epochs=100)

estimator.fit(input_fn=input_fn, steps=100)
print(estimator.evaluate(input_fn=input_fn))
