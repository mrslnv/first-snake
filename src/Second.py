from math import pi

def fib(n):
    if n < 3:
        return 1
    elif n == 3:
        return 2
    else:
        return fib(n-2)+fib(n-1)


for i in range(1,10):
    print(fib(i))

a = [str(round(pi,i)) for i in range(5,10)]
print(a)

ch=['a','b','c']
a = [(ch[i],ch[j]) for i in range(3) for j in range (3)]
print(a)

matrix = [
    [1,2,3,4],
    [2,3,4,5],
    [4,4,5,6]
]

x=[row[i]*2 for row in matrix for i in range(4)]
print(x)

x=[[row[i]*2 for i in range(4)]for row in matrix ]
print(x)