from random import randint
from math import floor

def generate_task(n, w, s, output_file):
    max_wi = floor(10 * w / n)
    max_si = floor(10 * s / n)
    w_arr = []
    s_arr = []
    c_arr = []
    w_sum = 0
    s_sum = 0

    while w_sum <= 2 * w or s_sum <= 2 * s:
        w_arr = []
        s_arr = []
        c_arr = []
        w_sum = 0
        s_sum = 0
        for i in range(n):
            wi = randint(1, max_wi)
            w_sum += wi
            w_arr.append(wi)

            si = randint(1, max_si)
            s_sum += si
            s_arr.append(si)

            c_arr.append(randint(1,n-1))

    with open(output_file, 'w') as f:
        f.write(f'{n}, {w}, {s}\n')
        for i in range(n):
            f.write(f'{w_arr[i]}, {s_arr[i]}, {c_arr[i]}\n')

generate_task(1001,10001,10001,'text.csv')
class Population:
    def __init__(self, n, w, s):
        pass


class Brain:
    def __init__(self):
        pass
