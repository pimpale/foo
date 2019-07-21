#!/bin/python3

x = 0
storage = 0
check = 0
currentnum = 0
precompi = []
precompo = []
precomplen = []
for number in range(1,1000000):
    if x >= check:
        storage = currentnum
        check = x
    x = 0
    currentnum = number
    while True:
        if number == 1:
            precompi.append(currentnum)
            precompo.append(number)
            precomplen.append(x)
            break
        elif number < currentnum:
            x = x+precomplen[int(number)-1]
            precompi.append(currentnum)
            precompo.append(number)
            precomplen.append(x)
            break
        elif number%2 == 0:
            number = number//2
            x += 1
        elif number%2 == 1:
            number = number*3+1
            x += 1
print(storage)
print(check)

