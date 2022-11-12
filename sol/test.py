from concurrent.futures import ThreadPoolExecutor
import test2 as t2

import test3 as t3


def do(i):
    t2.dosomethig("test"+str(i))

def do2(i):
    instance=t3.test3()
    instance.do("test"+str(i))


if __name__ == '__main__':
    

    with ThreadPoolExecutor() as executor:
        results = executor.map(do2, range(10))
        #wait for all results
        results = list(results)