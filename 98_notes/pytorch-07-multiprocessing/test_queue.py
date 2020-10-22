"""
    test Queue class

    task
    - I have a list of integers, I want to compute the square of each element by subprocesses,
      return the value, process id in order
"""

from multiprocessing import Process, Queue, current_process, cpu_count
import time


def square(a):
    return a**2


class QueueFunc:
    def __init__(self):
        pass

    def _func_queue(self, func, q_in, q_out, *args, **kwargs):
        """
        get data from q_in, execute func with *args & **kwargs, put process ids & results into q_out
        """
        while True:
            pos, var = q_in.get()
            if pos is None:
                break
            
            res = func(var, *args, **kwargs)
            q_out.put((pos, res))
        return

    def multiprocess_func(self, var, func, *args, **kwargs):
        """
        execute func in multiple processes
        """
        nprocs = cpu_count()

        processes = []
        q_in = Queue(1)
        q_out = Queue() 

        # 1) instantiate subprocesses
        for i in range(nprocs):
            p = Process(target=self._func_queue, args=tuple([func, q_in, q_out]), kwargs=kwargs)
            processes.append(p)
        # 2) set up & launch subprocesses
        for p in processes:
            p.daemon = True
            p.start()

        sent = [q_in.put((i, var[i])) for i in range(len(var))]
        # give each subprocess a terminate entry 'None, None' for the while loop to exit properly
        [q_in.put((None, None)) for _ in range(nprocs)]

        results = [[] for _ in range(len(var))]
        for i in range(len(sent)):
            index, res = q_out.get()
            results[index] = res
        # 3) wait for subprocesses to exit
        [p.join() for p in processes]

        return results


if __name__ == '__main__':
    a = list(range(0, cpu_count()*2))
    print(a)

    P = QueueFunc()
    results = P.multiprocess_func(a, square)
    print(results)
    
