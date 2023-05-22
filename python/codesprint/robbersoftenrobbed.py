import sys
import numpy as np

lines = sys.stdin.readlines()
K = int(lines[0].split()[1])
Kmid = K // 2
S = np.array([int(x) for x in lines[1].split()])

def solve_for_target_level(target):
    S_expanded = np.hstack([(S // target)*target, S % target])
    # top K entries, this represents all of mario's knapsacks
    S_sorted = np.sort(S_expanded)[::-1]
    bowser_gets = np.sum(S_sorted[:Kmid])
    mario_gets = np.sum(S_sorted[Kmid:])
    return bowser_gets, mario_gets

best_mario = 0
best_bowser = 0

for target in range(1, 1001):
    bowser_gets, mario_gets = solve_for_target_level(target)
    if mario_gets > best_mario:
        best_mario = mario_gets
        best_bowser = bowser_gets
    elif mario_gets == best_mario and bowser_gets < best_bowser:
        best_bowser = bowser_gets

print(best_mario, best_bowser)