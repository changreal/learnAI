# -*- coding:utf-8 -*-

from threading import Thread
from queue import Queue


class ThreadedGenerator(object):
    # daemon模式默认false，那么主线程会等到子线程全结束了再停，否则主线程一结束就关闭。
    def __init__(self, iterator, sentinel=object(), queue_maxsize=0, daemon=False):

        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator), # repr() 函数将对象转化为供解释器读取的形式。
            target=self._run     # run开始执行线程
        )
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self):
        self._started = False
        try:
            while True:
                self._queue.get(timeout=30)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    def __iter__(self):   # 迭代器
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration()
        return value


#  测试
def test():

    def gene():  # gene是i产生器
        i = 0
        while True:
            yield i
            i += 1

    t = gene()  # 得到i，那么t是一个生成器
    test = ThreadedGenerator(t)  # 第一个参数是t，生成器 可迭代对象

    for _ in range(10):
        print(next(test))  #next函数是自己新封装写的，调用。

    test.close()


if __name__ == '__main__':
    test()
