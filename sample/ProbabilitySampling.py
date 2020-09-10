import numpy as np


def get_input():
    words = input().split()
    times = [int(a) for a in input().split()]
    k = int(input())
    sum_times = sum(times)
    probs = [time / sum_times for time in times]
    return words, probs, k


def solution(words, probs, k):
    a = np.random.choice(words, k, p=probs)
    return a


words, probs, k = get_input()
print(solution(words, probs, k))
