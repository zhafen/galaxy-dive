#!/usr/bin/env python
'''Utilities for multiprocessing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import multiprocessing as mp
import os
import pdb
import sys
from types import MethodType

# Python 2/3 compatible copyreg
try:
    import copy_reg
except:
    import copyreg as copy_reg

import galaxy_dive.utils.utilities as utilities

########################################################################

def apply_among_processors( f, all_args, n_processors=mp.cpu_count() ):
  '''Takes a list of arguments and breaks it up and splits those chunks among the processors.
  Note: This currently does not return anything, so it doesn't work for functions where you want f to return something.
  However! It does work for shared memory objects, unlike Pool or parmap!

  Args:
    f (function) : The function to apply the args to.
    all_args (list) : Args to apply. Format, [ (args1), (args2), ... ]
    n_processors (int, optional) : Number of processors to use.
  '''

  def wrapped_f( args_chunk ):
    for args in args_chunk:
      f(*args)

  chunked_args = utilities.chunk_list( all_args, n_processors )

  ps = [ mp.Process( target=wrapped_f, args=(args_chunk,) ) for args_chunk in chunked_args ]

  [ p.start() for p in ps ]
  [ p.join() for p in ps ]

########################################################################

def mp_queue_to_list( queue, n_processors=mp.cpu_count() ):
  '''Convert a multiprocessing.Queue object to a list, using multiple processors to parse it.
  The list is unordered. It may also not work if the queue contains lists.

  Args:
    queue (mp.Queue) : The queue to turn into a list.
    n_processors (int) : Number of processors to use.
  '''

  def process_queue( q, l ):

    while True:

      l.acquire()
      if q.qsize() > 1:
        popped1 =  q.get()
        popped2 =  q.get()
        l.release()
      else:
        l.release()
        break

      if not isinstance( popped1, list ):
        popped1 = [ popped1, ]
      if not isinstance( popped2, list ):
        popped2 = [ popped2, ]
      
      q.put( popped1 + popped2 )

  lock = mp.Lock()

  proc = [ mp.Process( target=process_queue, args=(queue,lock) ) for _ in range( n_processors ) ]
  for p in proc:
      p.daemon = True
      p.start()
  [ p.join() for p in proc ]

  return queue.get()

########################################################################
'''The following is a version of Pool, written with classes in mind. It does not handle shared memory objects well.
https://stackoverflow.com/a/16071616
'''

def fun( f, q_in, q_out ):
  while True:
    i, x = q_in.get()
    if i is None:
      print( "PID {} finishing, PPID {}.".format( os.getpid(), os.getppid() ) )
      break
    q_out.put( (i, f( x )) )

def set_fun( f, q_in, q_out ):
  res_proc = set()
  while True:
    i, x = q_in.get()
    if i is None:
      print( "PID {} finishing, PPID {}.".format( os.getpid(), os.getppid() ) )
      q_out.put( res_proc )
      break
    res_proc = res_proc | f( x )

def parmap( f, X, n_processors=mp.cpu_count(), return_values=True, set_case=False, use_mp_queue_to_list=False ):
    '''Parallel map, viable with classes.

    Args:
      f (function) : Function to map to.
      X (list) : List of arguments to provide f
      n_processors (int) : Number of processors to use.
      return_values (bool) : If False, don't bother getting the results from the functions.
      set_case (bool) : If this option is True, it assumes that f returns a set, and that results should be the
        union of all those sets.
      use_mp_queue_to_list (bool) : Experimental. If True, try to use mp_queue_to_list to convert the list.
        Only works if set_case, currently.

    Returns:
      results (list or set) : The results.
    '''

    m = mp.Manager()

    q_in = m.Queue(1)
    q_out = m.Queue()

    if set_case:
      target_fun = set_fun
    else:
      target_fun = fun

    proc = [ mp.Process( target=target_fun, args=(f, q_in, q_out) )
            for _ in range( n_processors ) ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [ q_in.put( (i, x) ) for i, x in enumerate( X ) ]
    [ q_in.put( (None, None) ) for _ in range( n_processors ) ]

    print( "Getting results from queue. This could take a while..." )

    # Store the results
    if return_values:
      if set_case:

        if use_mp_queue_to_list:
          res = mp_queue_to_list( q_out, n_processors )
        else:
          res = [ q_out.get() for _ in range( n_processors ) ]

        [ p.join() for p in proc ]

        return res

      else:

        res = [ q_out.get() for _ in range( len( sent ) ) ]

        [ p.join() for p in proc ]

        return [ x for i, x in sorted( res ) ]

    else:
      [ p.join() for p in proc ]

      return

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

########################################################################

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    From https://stackoverflow.com/a/23654936
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
