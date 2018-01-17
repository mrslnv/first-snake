
print("start")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

im = mnist.train._images[0]
for i in range(28):
    for j in range(28):
        if im[i*28+j]:
            print(" ",end="")
        else:
            print("o",end="")
    print()
print("end")