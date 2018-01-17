x= [1,2]
y= [1,2]
print("dif",x[0]-y[0],x[1]-y[1])
i = 2
s = "a " * 2 + "," + "b " * 3
print("sdf " + s * i)
print("sdf",i*i)

word = "(THis is a word)"
print(word)
print(word[1:-1])

list = [1, 2, 3, 4, 5, 6]
print(list)
print(list[2:-2] * 2)

a, b = 0, 10
f = 1
while a < b:
    print("A",a)
    a,b = a+1,b-1
    f = f*(f+1)
    print("f",f)

a = 1
while a<10:
    if a:
        print("yes")
    if a>6:
        print("big big")
    elif a>3:
        print("big")
    a = a + 1
    if a > 9:
        print("end")

s = [0,0,0,0,0,0,0,0,0,0,0]
for i in range(5,10):
    s[i]=i
print(s)

a = 1
while i<10**2:
    a = a + i
    i = i + 1

print(a)

