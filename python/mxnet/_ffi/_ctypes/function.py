# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement, unused-import
"""
Function configuration API.
Acknowledgement: This file originates from incubator-tvm
"""
import ctypes
import traceback
from numbers import Number, Integral
import numpy as onp

from ...base import get_last_ffi_error, _LIB, check_call, py2cerror
from ..base import c_str
from .types import MXNetValue, TypeCode, MXNetPackedCFunc, MXNetCFuncFinalizer
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH
from ..._ctypes.ndarray import NDArrayBase
from .object import ObjectBase, PyNativeObject, _set_class_object
from . import object as _object

ObjectHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
MXRetValueHandle = ctypes.c_void_p

def _make_packed_func(handle, is_global):
    """Make a packed function class"""
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    obj.is_global = is_global
    obj.handle = handle
    return obj

def _get_global_func(name, allow_missing=False):
    handle = FunctionHandle()
    check_call(_LIB.MXNetFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return _make_packed_func(handle, False)

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)

def _ctypes_free_resource(rhandle):
    """callback to free resources when it it not needed."""
    pyobj = ctypes.cast(rhandle, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)


# Global callback that is always alive
MXNET_FREE_PYOBJ = MXNetCFuncFinalizer(_ctypes_free_resource)

def convert_to_mxnet_func(pyfunc):
    """Convert a python function to MXNET function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    mxnetfunc: mxnet.nd.Function
        The converted tvm function.
    """
    local_pyfunc = pyfunc

    def cfun(args, type_codes, num_args, ret, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = (C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args))
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception:
            msg = traceback.format_exc()
            msg = py2cerror(msg)
            _LIB.MXAPISetLastError(c_str(msg))
            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one return value")
            temp_args = []
            values, tcodes, _ = _make_mxnet_args((rv,), temp_args)
            if not isinstance(ret, MXRetValueHandle):
                ret = MXRetValueHandle(ret)
            if _LIB.MXCFuncSetReturn(ret, values, tcodes, ctypes.c_int(1)) != 0:
                raise get_last_ffi_error()
            _ = temp_args
            _ = rv
        return 0

    handle = FunctionHandle()
    f = MXNetPackedCFunc(cfun)

    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    if _LIB.MXFuncCreateFromCFunc(f, pyobj, MXNET_FREE_PYOBJ, ctypes.byref(handle)) != 0:
        raise get_last_ffi_error()
    return _make_packed_func(handle, False)

def _make_mxnet_args(args, temp_args):
    """Pack arguments into c args mxnet call accept"""
    num_args = len(args)
    values = (MXNetValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, NDArrayBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NDARRAYHANDLE
        elif isinstance(arg, ObjectBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, PyNativeObject):
            values[i].v_handle = arg.__mxnet_object__.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, str):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple, dict)):
            arg = _FUNC_CONVERT_TO_NODE(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        elif isinstance(arg, type):
            values[i].v_str = c_str(onp.dtype(arg).name)
            type_codes[i] = TypeCode.STR
        elif callable(arg):
            arg = convert_to_mxnet_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.PACKED_FUNC_HANDLE
            temp_args.append(arg)
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.MXNetFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_mxnet_args(args, temp_args)
        ret_val = MXNetValue()
        ret_tcode = ctypes.c_int()
        if _LIB.MXNetFuncCall(
                self.handle, values, tcodes, ctypes.c_int(num_args),
                ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return (RETURN_SWITCH[ret_tcode.value](ret_val) if ret_tcode.value != TypeCode.PYARG
                else RETURN_SWITCH[ret_tcode.value](ret_val, args))


def __init_handle_by_constructor__(fconstructor, args):
    """Initialize handle by constructor"""
    temp_args = []
    values, tcodes, num_args = _make_mxnet_args(args, temp_args)
    ret_val = MXNetValue()
    ret_tcode = ctypes.c_int()
    if _LIB.MXNetFuncCall(
            fconstructor.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
        raise get_last_ffi_error()
    _ = temp_args
    _ = args
    assert ret_tcode.value == TypeCode.OBJECT_HANDLE
    handle = ret_val.v_handle
    return handle

_object.__init_by_constructor__ = __init_handle_by_constructor__

_CLASS_PACKED_FUNC = None
_FUNC_CONVERT_TO_NODE = None

def _set_class_packed_func(packed_func_class):
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = packed_func_class

def _set_node_generic(func_convert_to_node):
    global _FUNC_CONVERT_TO_NODE
    _FUNC_CONVERT_TO_NODE = func_convert_to_node
