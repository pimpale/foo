#!/bin/python

import pandas as pd
import random
import sys

xs: list[float] = []
ys: list[float] = []
zs: list[float] = []
temps: list[float] = []
absmags: list[float] = []

for i in range(20):
	xs.append(random.random()*100 - 50)
	ys.append(random.random()*100 - 50)
	zs.append(random.random()*100 - 50)
	temps.append(random.random()*8000)
	absmags.append(random.random()*10 - 5)


df = pd.DataFrame(data={'x': xs, 'y': ys, 'z': zs, 'temp':zs, 'absmag': absmags})

# save to the destination
df.to_csv(sys.argv[1], index_label='rowid')
