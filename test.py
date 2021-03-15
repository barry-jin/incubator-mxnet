import matplotlib.pyplot as plt
import numpy as onp

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'NDArray', 'Integer', 'Object', 'Container', 'Str', 'None'
# sizes = [17.32706514, 22.02820685, 24.5802552, 11.28274009, 21.55809268, 3.223640027]
# explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# labels = 'NDArray', 'None', 'object', 'Integer'
# sizes = [87.56756757, 8.108108108, 3.243243243, 1.081081081]
# explode = (0.1, 0, 0, 0)

# fig1, ax1 = plt.subplots()
# # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
# #         shadow=True, startangle=90)
# patches, texts, auto = ax1.pie(sizes, explode=explode, autopct='%1.1f%%',
#                                shadow=True, startangle=90)
# ax1.legend(patches, labels, loc="best")
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()
# def AddOne():
#     pass
from mxnet._ctypes import _api_internal
from mxnet import np, npx 
npx.set_np()

# from mxnet._ffi.function import _init_api

def addone(a):
    # AddOne has been registered via `_ini_api` call below

    return _api_internal.AddOne(a)
a = [np.zeros((2,2)) for _ in range(5)]
print(addone(a))
# addone(['1', '2', '3'])

# _init_api("mxnet.test")


# # addone({'a': 1})

# # addone([np.zeros((2,2))])

# addone('1')

# addone([np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))])


# addone({'a': np.zeros((2,2))})


# print("""
# Traceback (most recent call last):
#   File "test.py", line 7, in <module>
#     addone({'a': 1})
#   File "test.py", line 5, in addone
#     return AddOne(a)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 195, in __call__
#     values, tcodes, num_args = _make_mxnet_args(args, temp_args)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 161, in _make_mxnet_args
#     raise TypeError("Don't know how to handle type %s" % type(arg))
# TypeError: Don't know how to handle type <class 'dict'>
# """)


# print("""
# Traceback (most recent call last):
#   File "test.py", line 7, in <module>
#     addone([np.zeros((2,2))])
#   File "test.py", line 5, in addone
#     return _api_internal.AddOne(a)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 195, in __call__
#     values, tcodes, num_args = _make_mxnet_args(args, temp_args)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 145, in _make_mxnet_args
#     arg = _FUNC_CONVERT_TO_NODE(arg)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 65, in convert_to_node
#     value = [convert_to_node(x) for x in value]
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 65, in <listcomp>
#     value = [convert_to_node(x) for x in value]
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 76, in convert_to_node
#     raise ValueError("don't know how to convert type %s to node" % type(value))
# ValueError: don't know how to convert type <class 'mxnet.numpy.ndarray'> to node
# """)

# print("""
# Traceback (most recent call last):
#   File "test.py", line 7, in <module>
#     addone('1')
#   File "test.py", line 5, in addone
#     return _api_internal.AddOne(a)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 201, in __call__
#     raise get_last_ffi_error()
# mxnet.base.MXNetError: Traceback (most recent call last):
#   File "../include/mxnet/runtime/packed_func.h", line 428
# MXNetError: Check failed: type_code_ == kDLInt (8 vs. 0) : expected int but get str
# """)

# print("""
# Traceback (most recent call last):
#   File "test.py", line 34, in <module>
#     addone(['1', '2', '3'])
#   File "test.py", line 32, in addone
#     return _api_internal.AddOne(a)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 195, in __call__
#     values, tcodes, num_args = _make_mxnet_args(args, temp_args)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/_ctypes/function.py", line 145, in _make_mxnet_args
#     arg = _FUNC_CONVERT_TO_NODE(arg)
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 65, in convert_to_node
#     value = [convert_to_node(x) for x in value]
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 65, in <listcomp>
#     value = [convert_to_node(x) for x in value]
#   File "/Users/zhenghuj/GitHub/incubator-mxnet/python/mxnet/_ffi/node_generic.py", line 76, in convert_to_node
#     raise ValueError("don't know how to convert type %s to node" % type(value))
# ValueError: don't know how to convert type <class 'str'> to node
# """)


