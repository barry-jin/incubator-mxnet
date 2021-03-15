import timeit
setup = """
from mxnet.base import c_handle_array
from mxnet import np, npx
from mxnet import _api_internal
import ctypes

npx.set_np()

args = []
for i in range(1000):
    args.append(np.zeros((10000,10000)))

"""

stmt = """
# arr = c_handle_array(args)
# hdl = ctypes.cast(c_handle_array(args), ctypes.c_void_p)
_api_internal._nop(*args)
"""

timer = timeit.Timer(setup=setup,
                     stmt=stmt)
num_repeat = 1000
print(timer.timeit(num_repeat) / num_repeat)
