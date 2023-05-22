import sys

lines = sys.stdin.readlines()

possible = True
ngoombas = 0
for l in lines[1:]:
    gi, bi = l.split()
    gi = int(gi)
    bi = int(bi)
    ngoombas += gi
    if ngoombas < bi:
        possible = False
        break

if possible:
    print("possible")
else:
    print("impossible")