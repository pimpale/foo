import sys
import numpy as np

lines = sys.stdin.readlines()[1:]
fullyexplored = np.zeros((len(lines), 3))

positions = []

for i, c in enumerate(lines[0]):
    if c == '.':
        positions.append((0,i))

possible = False

while len(positions) != 0:
    y, x = positions.pop()
    if fullyexplored[y, x]:
        continue
    else:
        fullyexplored[y, x] = 1
    if y >= len(lines)-1:
        possible = True
        break
    
    current_val = lines[y][x]
    if current_val == '/':
        positions.append((y+1,x))
    if current_val == '.':
        if x > 0:
            if lines[y][x-1] == '.':
                positions.append((y, x-1))
        if x < 2:
            if lines[y][x+1] == '.':
                positions.append((y, x+1))
        if lines[y+1][x] == '/' or lines[y+1][x] == '.':
            positions.append((y+1, x))
    if current_val == '*':
        if x > 0:
            if lines[y][x-1] == '*':
                positions.append((y, x-1))
        if x < 2:
            if lines[y][x+1] == '*':
                positions.append((y, x+1))
        positions.append((y+1, x))


if possible:
    print('YES')
else:
    print('NO')


    