
def get_collatz_steps(z: int) -> int:
    n = 0
    while z != 1:
        if z % 2 == 0:
            z //= 2
        else:
            z = 3*z + 1
        n += 1

    return n

print(get_collatz_steps(1))
print(get_collatz_steps(13))
