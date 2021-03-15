import timeit

setup = """
import ctypes

from mxnet.base import _LIB
from mxnet.base import c_handle_array
from mxnet.base import NDArrayHandle, CachedOpHandle, SymbolHandle
from mxnet.base import check_call
from mxnet import _global_var
from mxnet import np, npx
npx.set_np()

def call_cached_op(*args, **kwargs):
    is_np_sym = True
    out = kwargs.pop('out', None)
    default_ctx = kwargs.pop('default_ctx', None)
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
    if kwargs:
        raise TypeError(
            "CachedOp.__call__ got unexpected keyword argument(s): " + \
            ', '.join(kwargs.keys()))

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
        ctypes.c_void_p(),
        ctypes.c_int(len(args)),
        c_handle_array(args),
        ctypes.c_int(default_ctx.device_typeid),
        ctypes.c_int(default_ctx.device_id),
        ctypes.byref(num_output),
        ctypes.byref(output_vars),
        ctypes.byref(out_stypes)))

        # ctypes.c_void_p()
        # ctypes.c_int(len(args))
        # c_handle_array(args)
        # ctypes.c_int(default_ctx.device_typeid)
        # ctypes.c_int(default_ctx.device_id)
        # ctypes.byref(num_output)
        # ctypes.byref(output_vars)
        # ctypes.byref(out_stypes)

    if original_output is not None:
        return original_output
    create_ndarray_fn = _global_var._np_ndarray_cls if is_np_sym else _global_var._ndarray_cls
    if num_output.value == 1:
        return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle),
                                    stype=out_stypes[0])
    else:
        return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                    stype=out_stypes[i]) for i in range(num_output.value)]

inputs = []
for i in range(3):
    inputs.append(np.zeros((2,2)))
"""

setup1 = """

from mxnet._cy3.ndarray import call_cached_op
from mxnet import np, npx
npx.set_np()

inputs = []
for i in range(3):
    inputs.append(np.zeros((2,2)))
"""

setup2 = """
import ctypes
from mxnet.ndarray._internal import NDArrayBase
from mxnet import _api_internal
from mxnet import np, npx
npx.set_np()

def call_cached_op(*args, **kwargs):
    default_ctx = kwargs.pop('default_ctx', None)
    out = kwargs.pop('out', None)
    if kwargs:
        raise TypeError(
            "CachedOp.__call__ got unexpected keyword argument(s): " + \
            ', '.join(kwargs.keys()))
    if True:
        if len(args) == 1 and args[0] is None:
            args = []
        type_id = default_ctx.device_typeid if default_ctx else None
        device_id = default_ctx.device_id if default_ctx else None
        out_arg = out if out is not None and not isinstance(out, NDArrayBase) else (out, )
        output_vars = _api_internal._nop(
            ctypes.c_void_p(),
            len(args),
            *args,
            type_id,
            device_id,
            *out_arg
        )
        return
        if out is not None:
            return out
        if isinstance(output_vars, NDArrayBase):
            return output_vars
        else:
            return list(output_vars)

inputs = []
for i in range(3):
    inputs.append(np.zeros((2,2)))
"""


stmt = """
call_cached_op(*inputs)
"""

timer = timeit.Timer(setup=setup,
                     stmt=stmt)
num_repeat = 1000
print(timer.timeit(num_repeat) / num_repeat)
