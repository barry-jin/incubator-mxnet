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
# pylint: disable=invalid-name, protected-access, too-many-arguments
# pylint: disable=global-statement, unused-import
"""CachedOp API."""

import ctypes

from ..base import _LIB
from ..base import c_handle_array
from ..base import NDArrayHandle, CachedOpHandle, SymbolHandle
from ..base import check_call
from .. import _global_var
from ..ndarray._internal import NDArrayBase
from . import _api_internal

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, opr_name, array, _):
        """ ctypes function """
        callback(name, opr_name, array)
    return callback_handle

class CachedOp(object):
    """Cached operator handle."""
    __slots__ = ["handle", "is_np_sym", "_monitor_callback"]

    def __init__(self, sym, flags=(), thread_safe=False):
        # self.handle = CachedOpHandle()
        self._monitor_callback = None

        from ..symbol.numpy._symbol import _Symbol
        self.is_np_sym = bool(isinstance(sym, _Symbol))

        # check_call(_LIB.MXCreateCachedOp(
        #     sym.handle,
        #     len(flags),
        #     c_str_array([key for key, _ in flags]),
        #     c_str_array([str(val) for _, val in flags]),
        #     ctypes.byref(self.handle),
        #     ctypes.c_bool(thread_safe)))
        
        flags = {key: str(value) for key, value in flags}
        self.handle = CachedOpHandle(_api_internal.create(
            sym.handle,
            len(flags),
            flags,
            thread_safe
        ))

    def __del__(self):
        # check_call(_LIB.MXFreeCachedOp(self.handle))
        _api_internal.free(self.handle)

    def get_optimized_symbol(self):
        """Get an optimized version of the symbol from the cached op.

        Returns
        -------
        symbol : Symbol
            Optimized symbol from the executor.
        """
        from ..symbol import Symbol
        # sym_handle = SymbolHandle()
        # check_call(_LIB.MXCachedOpGetOptimizedSymbol(self.handle, ctypes.byref(sym_handle)))
        sym_handle = SymbolHandle(_api_internal.get_optimized_symbol(self.handle))
        ret = Symbol(sym_handle)
        return ret

    def __call__(self, *args, **kwargs):
        """ctypes implementation of imperative invoke wrapper"""
        # New FFI only supports numpy ndarray
        out = kwargs.pop('out', None)
        default_ctx = kwargs.pop('default_ctx', None)
        if kwargs:
            raise TypeError(
                "CachedOp.__call__ got unexpected keyword argument(s): " + \
                ', '.join(kwargs.keys()))
        if self.is_np_sym:
            if len(args) == 1 and args[0] is None:
                args = []
            output_vars = _api_internal.invoke(
                self.handle,
                args,
                out if isinstance(out, NDArrayBase) or out is None else (out, ),
                default_ctx.device_typeid if default_ctx else None,
                default_ctx.device_id if default_ctx else None
            )
            if out is not None:
                return out
            if len(output_vars) == 1:
                return output_vars[0]
            else:
                return list(output_vars)
        else:
            if out is not None:
                original_output = out
                if isinstance(out, NDArrayBase):
                    out = (out,)
                num_output = ctypes.c_int(len(out))
                output_vars = c_handle_array(out)
                output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
            else:
                original_output = None
                output_vars = ctypes.POINTER(NDArrayHandle)()
                num_output = ctypes.c_int(0)

            # return output stypes to avoid the c_api call for checking
            # a handle's stype in _ndarray_cls
            out_stypes = ctypes.POINTER(ctypes.c_int)()

            # (None, ) -> []
            if len(args) == 1 and args[0] is None:
                args = []
                assert default_ctx is not None, 'default_ctx is required if no input is provided'
            else:
                default_ctx = args[0].ctx if default_ctx is None else default_ctx

            check_call(_LIB.MXInvokeCachedOp(
                self.handle,
                ctypes.c_int(len(args)),
                c_handle_array(args),
                ctypes.c_int(default_ctx.device_typeid),
                ctypes.c_int(default_ctx.device_id),
                ctypes.byref(num_output),
                ctypes.byref(output_vars),
                ctypes.byref(out_stypes)))

            if original_output is not None:
                return original_output
            create_ndarray_fn = _global_var._ndarray_cls
            if num_output.value == 1:
                return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle),
                                        stype=out_stypes[0])
            else:
                return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                        stype=out_stypes[i]) for i in range(num_output.value)]

    def _register_op_hook(self, callback, monitor_all=False):
        """Install callback for monitor.

        Parameters
        ----------
        callback : function
            Takes a string for node_name, string for op_name and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input _imperative_invoked output, otherwise monitor output only.
        """
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
        if callback:
            self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
        # check_call(_LIB.MXCachedOpRegisterOpHook(
        #     self.handle,
        #     self._monitor_callback,
        #     ctypes.c_int(monitor_all)))
        callback_ptr = ctypes.cast(self._monitor_callback, ctypes.c_void_p)
        _api_internal.register_op_hook(self.handle, callback_ptr, monitor_all)
