#!/usr/bin/env python
'''Utilities for multiprocessing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult

import copy_reg
from types import MethodType

########################################################################

def _pickle_method(method):
  '''The majority of this was taken from the following StackOverflow answer:
  http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma/7309686#7309686
  '''

  func_name = method.im_func.__name__
  obj = method.im_self
  cls = method.im_class
  return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
  '''The majority of this was taken from the following StackOverflow answer:
  http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma/7309686#7309686
  '''

  for cls in cls.mro():
    try:
      func = cls.__dict__[func_name]
    except KeyError:
      pass
    else:
      break
  return func.__get__(obj, cls)

def make_classes_picklable():
  '''The majority of this was taken from the following StackOverflow answer:
  http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma/7309686#7309686
  '''
  copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)

########################################################################
'''The following is from another stack exchange answer.
https://stackoverflow.com/a/16071616
'''
import multiprocessing


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):

    m = multiprocessing.Manager()

    q_in = m.Queue(1)
    q_out = m.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


if __name__ == '__main__':
    print(parmap(lambda i: i * 2, [1, 2, 3, 4, 6, 7, 8]))
