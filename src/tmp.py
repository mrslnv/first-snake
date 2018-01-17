import numpy as np

x, y = np.meshgrid(np.arange(1,4),np.arange(5,9))
print("x ",x)
print("y ",y)

print("x ravel",x.ravel())
print("y ravel",y.ravel())

z = np.c_[x.ravel(),y.ravel()]
print(z)

sampler = np.random.choice(np.arange(100), 50, replace=False)
a = np.arange(1000,900,-1)
b = np.array([i for i in a])
a[sampler]
b[sampler]

def sample(ratio, *arrays):
    ret = []
    a = arrays[0]
    l = len(a)
    sampler = np.random.choice(np.arange(l), int(round(l*ratio)), replace=False)
    for a in arrays:
        ret.append(a[sampler])
    return ret


x,y = sample(0.5,np.arange(1,10),np.arange(10,20))
print(x)
print(y)