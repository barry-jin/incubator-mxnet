import mxnet as mx
import struct
import timeit
from mxnet.numpy.multiarray import transpose
import numpy as _np
from collections import namedtuple
from mxnet import np, npx 
npx.set_np()


setup = """
from mxnet import np, npx
npx.set_np()
inp = np.zeros((10, 2))
"""
stmt = """
out = npx.arange_like(inp, start=0.0, step=1.0, repeat=1, axis=None)
"""
timer = timeit.Timer(setup=setup,
                     stmt=stmt)
num_repeat = 1000
# print(min(timer.repeat(repeat=10, number=num_repeat)) / num_repeat)


inp = np.zeros((10, 2))
out = npx.arange_like(inp, start=0.0, step=1.0, repeat=1, axis=None)

# print(out)
