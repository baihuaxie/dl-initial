"""
    test Pipe objects
"""

import multiprocessing as mp


def worker(child):
    """
    """
    print(child.recv())
    child.send('this is from a child')

class School():

    def __init__(self):

        self.teacher, self.child = mp.Pipe()
        self.p = mp.Process(target=worker, args=(self.child,))
        self.p.start()

        self.teacher.send('this is from a teacher')
        print(self.teacher.recv())


if __name__ == '__main__':
    school = School()
