"""
    examples of multiprocessing in pytorch
"""
import pytest
import time
from multiprocessing import Process, Pool

# https://towardsdatascience.com/a-hands-on-guide-to-multiprocessing-in-python-48b59bfcc89e

def is_prime(n):
    """ check if n is prime number """
    if (n <= 1):
        return 'not a prime number'
    if (n <= 3):
        return 'a prime number'

    # check from 2 to n-1
    for i in range(2, n):
        if (n % i == 0):
            return 'not a prime number'

    return 'a prime number'


def func(x):
    time.sleep(0.1)
    # Q: why is this not printed?
    print("{} is {}".format(x, is_prime(x)))


def test_single_process():
    """ test single process """
    start_time = time.time()
    for i in range(1, 20):
        func(i)
    print("Time taken = {} seconds".format(time.time() - start_time))
    print()


def test_process():
    """ test Process class """
    start_time = time.time()
    processes = []
    for i in range(1, 20):
        p = Process(target=func, args=(i,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print("Time taken = {} seconds".format(time.time() - start_time))
    print()


def test_pool():
    """ test Pool class """
    start_time = time.time()
    pool = Pool()
    pool.map(func, range(1, 20))
    pool.close()
    print("Time taken = {} seconds".format(time.time() - start_time))
    print()