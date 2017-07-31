#!/usr/bin/env python
'''Utilities for multiprocessing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy_reg
import multiprocessing as mp
from types import MethodType

import galaxy_diver.utils.utilities as utilities

########################################################################

def apply_among_processors( f, all_args, n_procs=mp.cpu_count() ):
  '''Takes a list of arguments and breaks it up and splits those chunks among the processors.
  Note: This currently does not return anything, so it doesn't work for functions where you want f to return something.
  However! It does work for shared memory objects, unlike Pool or parmap!

  Args:
    f (function) : The function to apply the args to.
    all_args (list) : Args to apply. Format, [ (args1), (args2), ... ]
    n_procs (int, optional) : Number of processors to use.
  '''

  def wrapped_f( args_chunk ):
    for args in args_chunk:
      f(*args)

  chunked_args = utilities.chunk_list( all_args, n_procs )

  ps = [ mp.Process( target=wrapped_f, args=(args_chunk,) ) for args_chunk in chunked_args ]

  [ p.start() for p in ps ]
  [ p.join() for p in ps ]

########################################################################
'''The following is a version of Pool, written with classes in mind. It does not handle shared memory objects well.
https://stackoverflow.com/a/16071616
'''

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=mp.cpu_count()):

    m = mp.Manager()

    q_in = m.Queue(1)
    q_out = m.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
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

########################################################################
'''This section contains efforts to make classes pickleable, allowing multiprocessing.Pool to be used.'''

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
