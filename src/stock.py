s = "http://akcie-cz.kurzy.cz/emise/"
f = open('stock.data', 'r')

for line in f:
    print(s+line[:-1]+"/")