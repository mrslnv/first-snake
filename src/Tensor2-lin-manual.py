import tensorflow as tf
import copy

def run(batchSize,iterations):
    W = tf.Variable([.1],tf.float32)
    b = tf.Variable([.1],tf.float32)

    x = tf.placeholder(tf.float32)

    model = W * x + b

    y = tf.placeholder(tf.float32)

    optimizer = tf.train.GradientDescentOptimizer(0.0137)
    loss = tf.reduce_sum(tf.sqrt(tf.square(model - y)))
    train = optimizer.minimize(loss)

    def fce(x):
        return 3*x+8

    x_in = [i for i in range(batchSize)]
    y_out = [fce(xi) for xi in x_in]

    print("x",x_in)
    print("y",y_out)

    init = tf.global_variables_initializer()
    ses = tf.Session()
    ses.run(init)
    ses.run(init)

    for i in range(iterations):
        if (not i % 1000): print("Train",i)
        ses.run(train,{x:x_in, y:y_out})

    print("Train done")
    Wc, bc, lossc = ses.run([W,b,loss],{x:x_in, y:y_out})
    print("",Wc,"* x +",bc)
    print("loss",lossc)

    ses.close()

    return Wc, bc, lossc

iter = [10,100,1000]
batch = [2,4,10,100,1000]

res = []
param = []
for it in iter:
    for b in batch:
        res.append(copy.deepcopy(run(b,it)))
        param.append((b,it))

for r,p in zip(res,param):
    print(p,"-",r)