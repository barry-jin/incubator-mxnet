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

"""Register backend ops in mxnet.ndarray namespace"""
import os as _os
import ctypes
import numpy as _np  # pylint: disable=unused-import

from ._internal import NDArrayBase, _imperative_invoke # pylint: disable=unused-import
from ..ndarray_doc import _build_doc

from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null, _is_np_op, _output_is_list  # pylint: disable=unused-import
from ..util import use_np_shape  # pylint: disable=unused-import
from .._ctypes import _api_internal  # pylint: disable=unused-import

rest_ops_set = {'multibox_prior', 'rsubtract_scalar', 'random_flip_top_bottom', 'instance_norm', 'matrix_rank_none_tol',
                'logical_or_scalar', 'rnn_param_concat', 'normalize', 'amp_multicast', 'pick', 'bitwise_xor_scalar',
                'rpower_scalar', 'pinv_scalar_rcond', 'multibox_detection', 'smooth_l1', 'power_scalar', 'rldexp_scalar',
                'multi_all_finite', 'sldwin_atten_context', 'powerd', 'slice_assign_scalar', 'insert_tensor', 'lcm_scalar',
                'random_color_jitter', 'crop', 'rfmod_scalar', 'arange_like', 'sometrue', 'Custom', 'bitwise_or_scalar',
                'ldexp_scalar', 'less_equal_scalar', 'where_rscalar', 'one_hot', 'minimum_scalar', 'all_finite',
                'rtrue_divide_scalar', 'intgemm_prepare_weight', 'not_equal_scalar', 'activation', 'scatter_set_nd',
                'logical_and_scalar', 'multibox_target', 'masked_softmax', 'sldwin_atten_score', 'equal_scalar', 'cvimresize',
                'boolean_mask_assign_tensor', 'random_lighting', 'bitwise_and_scalar', 'true_divide_scalar', 'insert_scalar',
                'embedding', 'greater_scalar', 'contrib_quantize', 'logical_xor_scalar', 'arctan2_scalar', 'constraint_check',
                'random_randint', 'fmin_scalar', 'less_scalar', 'rnn', 'convolution', 'bipartite_matching', 'scalar_poisson',
                'roi_pooling', 'random_resized_crop', 'fmax_scalar', 'masked_log_softmax', 'slice_assign', 'pooling',
                'maximum_scalar', 'box_iou', 'rmod_scalar', 'tensordot_int_axes', 'contrib_calibrate_entropy', 'softmax',
                'fully_connected', 'log_softmax', 'copysign_scalar', 'normal_n', 'random_hue', 'slice', 'topk', 'random_saturation',
                'intgemm_fully_connected', 'cvimread', 'rcopysign_scalar', 'gcd_scalar', 'add_scalar', 'batch_norm', 'box_decode',
                'modulated_deformable_convolution', 'categorical', 'bernoulli', 'multiply_scalar', 'leaky_relu', 'batch_dot',
                'sldwin_atten_mask_like', 'fmod_scalar', 'greater_equal_scalar', 'random_brightness', 'mod_scalar', 'deconvolution',
                'random_crop', 'cvimdecode', 'tensor_poisson', 'layer_norm', 'insert_slice', 'where_scalar2', 'ctc_loss',
                'random_contrast', 'hypot_scalar', 'product', 'reshape_like', 'where_lscalar', 'uniform_n', 'boolean_mask_assign_scalar',
                'subtract_scalar', 'repeats', 'contrib_quantize_v2', 'adjust_lighting', 'broadcast_like', 'sequence_mask', 'dropout',
                'deformable_convolution', 'random_flip_left_right', 'box_nms', 'boolean_mask', 'rarctan2_scalar'}

def _verify_all_np_ndarrays(op_name, func_name, args, out):
    """Verify if all the arrays are numpy ndarrays.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    args : list of arrays
        Input ndarray arguments to be checked.
    out : ndarray or None or list of ndarrays
        User-provided output ndarrays.
    """
    from ..numpy import ndarray as np_ndarray
    for arr in args:
        if (arr is not None) and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a numpy operator which can only accept '
                            'MXNet numpy ndarrays, while received a legacy ndarray. '
                            'Please ensure that you have activated numpy semantics by calling '
                            '`npx.set_np()` in your code. If you still see this error with numpy '
                            'semantics activated, please call `as_np_ndarray()` upon the legacy '
                            'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                            'converted array to this operator.'
                            .format(op_name, func_name))
    if out is None:
        return
    if not isinstance(out, (list, tuple)):
        out = [out]
    for arr in out:
        if (arr is not None) and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a numpy operator which can only accept '
                            'MXNet numpy ndarrays, while received a legacy ndarray. '
                            'Please ensure that you have activated numpy semantics by calling '
                            '`npx.set_np()` in your code. If you still see this error with numpy '
                            'semantics activated, please call `as_np_ndarray()` upon the legacy '
                            'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                            'converted array to this operator.'
                            .format(op_name, func_name))


def _verify_all_legacy_ndarrays(op_name, func_name, args, out):
    """Verify if all the arrays are legacy ndarrays.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    args : list of arrays
        Input ndarray arguments to be checked.
    out : ndarray or None or list of ndarrays
        User-provided output ndarrays.
    """
    from ..numpy import ndarray as np_ndarray
    for arr in args:
        if (arr is not None) and (isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a legacy operator which can only accept '
                            'legacy ndarrays, while received an MXNet numpy ndarray. '
                            'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                            'convert it to a legacy ndarray, and then feed the converted '
                            'array to this operator.'
                            .format(op_name, func_name))
    if out is None:
        return
    if not isinstance(out, (list, tuple)):
        out = [out]
    for arr in out:
        if (arr is not None) and (isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a legacy operator which can only write to '
                            'legacy ndarrays, while received an MXNet numpy ndarray. '
                            'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                            'convert it to a legacy ndarray, and then feed the converted '
                            'array to this operator.'
                            .format(op_name, func_name))


def _np_imperative_invoke(handle, ndargs, out):
    """PackedFunc based numpy operator invocation call"""
    output_vars = _api_internal.invoke(handle, *ndargs, out)
    if out is not None:
        return out
    if isinstance(output_vars, NDArrayBase):
        return output_vars
    else:
        return list(output_vars)


# def np_imperative_invoke_params(handle, ndargs, params, out):
#     """PackedFunc based numpy operator invocation call"""
#     keys, vals = params.keys(), [str(val) for val in params.values()]
#     output_vars = _api_internal.invoke_with_params(handle, len(ndargs), *ndargs, *keys, *vals, out)
#     if out is not None:
#         return out
#     if isinstance(output_vars, NDArrayBase):
#         return output_vars
#     else:
#         return list(output_vars)


# pylint: disable=too-many-locals
def _generate_ndarray_function_code(handle, op_name, func_name, signature_only=False):
    """Generate function for ndarray op by handle and function op_name."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(op_name,
                         py_str(desc.value),
                         arg_names,
                         arg_types,
                         [py_str(arg_descs[i]) for i in range(narg)],
                         key_var_num_args,
                         ret_type)

    dtype_name = None
    arr_name = None
    ndsignature = []
    signature = []
    ndarg_names = []
    kwarg_names = []
    for i in range(narg):
        name, atype = arg_names[i], arg_types[i]
        if name == 'dtype':
            dtype_name = name
            signature.append('%s=_Null'%name)
        elif atype.startswith('NDArray') or atype.startswith('Symbol'):
            assert not arr_name, \
                "Op can only have one argument with variable " \
                "size and it must be the last argument."
            if atype.endswith('[]'):
                ndsignature.append('*%s'%name)
                arr_name = name
            else:
                ndsignature.append('%s=None'%name)
                ndarg_names.append(name)
        else:
            signature.append('%s=_Null'%name)
            kwarg_names.append(name)
    signature.append('out=None')
    signature.append('name=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    code = []
    is_np_op = _is_np_op(op_name)
    output_is_list = _output_is_list(op_name)
    if is_np_op:
        if arr_name:
            code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
            if not signature_only:
                code.append("""
    ndargs = []
    for i in {}:
        assert isinstance(i, NDArrayBase), \\
            "Positional arguments must have NDArray type, " \\
            "but got %s"%str(i)
        ndargs.append(i)""".format(arr_name))
                if dtype_name is not None:
                    code.append("""
    if '%s' in kwargs:
        if _np.dtype(kwargs['%s']).names:
            kwargs['%s'] = _np.dtype(kwargs['%s']).names[0]
        else:
            kwargs['%s'] = _np.dtype(kwargs['%s']).name """%(
                dtype_name, dtype_name, dtype_name, dtype_name, dtype_name, dtype_name))
                code.append("""
    _ = kwargs.pop('name', None)
    out = kwargs.pop('out', None)""")
            if not signature_only:
                code.append("""
    _verify_all_np_ndarrays("{op_name}", "{func_name}", ndargs, out)
            """.format(op_name=op_name, func_name=func_name))
                code.append("""
    return _imperative_invoke(%d, ndargs, kwargs.keys(), kwargs.values(), out, True, %s)"""%(
        handle.value, str(output_is_list)))
    # return np_imperative_invoke_params(%d, ndargs, kwargs, out)"""%(
    #     handle.value))
            else:
                code.append("""
    return (0,)""")
        else:
            code.append("""
def %s(%s):"""%(func_name, ', '.join(signature)))
            if not signature_only:
                code.append("""
    ndargs = []""")
                # NDArray args
                for name in ndarg_names: # pylint: disable=redefined-argument-from-local
                    code.append("""
    if {name} is not None:
        assert isinstance({name}, NDArrayBase), \\
            "Argument {name} must have NDArray type, but got %s"%str({name})
        ndargs.append({name})""".format(name=name))
                # kwargs
                if not kwarg_names:
    
    #                 if dtype_name is not None:
    #                     code.append("""
    # if %s is not _Null and %s is not None:
    #     kwargs['%s'] = _np.dtype(%s).name"""%(dtype_name, dtype_name, dtype_name, dtype_name))

                    code.append("""
    _verify_all_np_ndarrays("{op_name}", "{func_name}", ndargs, out)
            """.format(op_name=op_name, func_name=func_name))
                    if not signature_only:
                        code.append("""
    return _np_imperative_invoke(%d, ndargs, out)"""%(handle.value))
    #                     code.append("""
    # return _imperative_invoke(%d, ndargs, kwargs.keys(), kwargs.values(), out, True, False)"""%(handle.value))
                    else:
                        code.append("""
    return (0,)""")
                else:
                    for name in kwarg_names: # pylint: disable=redefined-argument-from-local
                        code.append("""
    if %s is not _Null:
        kwargs['%s'] = %s"""%(name, name, name))
                # dtype
                    if dtype_name is not None:
                        code.append("""
    if %s is not _Null and %s is not None:
        kwargs['%s'] = _np.dtype(%s).name"""%(dtype_name, dtype_name, dtype_name, dtype_name))
                    if not signature_only:
                        code.append("""
    _verify_all_np_ndarrays("{op_name}", "{func_name}", ndargs, out)
            """.format(op_name=op_name, func_name=func_name))
                        code.append("""
    return _imperative_invoke(%d, ndargs, kwargs.keys(), kwargs.values(), out, True, %s)"""%(
        handle.value, str(output_is_list)))
    # return np_imperative_invoke_params(%d, ndargs, kwargs, out)"""%(
    #     handle.value))
                    else:
                        code.append("""
    return (0,)""")


    else:
        if arr_name:
            code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
            if not signature_only:
                code.append("""
    ndargs = []
    for i in {}:
        assert isinstance(i, NDArrayBase), \\
            "Positional arguments must have NDArray type, " \\
            "but got %s"%str(i)
        ndargs.append(i)""".format(arr_name))
                if dtype_name is not None:
                    code.append("""
    if '%s' in kwargs:
        if _np.dtype(kwargs['%s']).names:
            kwargs['%s'] = _np.dtype(kwargs['%s']).names[0]
        else:
            kwargs['%s'] = _np.dtype(kwargs['%s']).name """%(
                dtype_name, dtype_name, dtype_name, dtype_name, dtype_name, dtype_name))
                code.append("""
    _ = kwargs.pop('name', None)
    out = kwargs.pop('out', None)
    keys = list(kwargs.keys())
    vals = list(kwargs.values())""")
        else:
            code.append("""
def %s(%s):"""%(func_name, ', '.join(signature)))
            if not signature_only:
                code.append("""
    ndargs = []
    keys = list(kwargs.keys())
    vals = list(kwargs.values())""")
                # NDArray args
                for name in ndarg_names: # pylint: disable=redefined-argument-from-local
                    code.append("""
    if {name} is not None:
        assert isinstance({name}, NDArrayBase), \\
            "Argument {name} must have NDArray type, but got %s"%str({name})
        ndargs.append({name})""".format(name=name))
                # kwargs
                for name in kwarg_names: # pylint: disable=redefined-argument-from-local
                    code.append("""
    if %s is not _Null:
        keys.append('%s')
        vals.append(%s)"""%(name, name, name))
                # dtype
                if dtype_name is not None:
                    code.append("""
    if %s is not _Null:
        keys.append('%s')
        if _np.dtype(%s).names:
            vals.append(_np.dtype(%s).names[0])
        else:
            vals.append(_np.dtype(%s).name) """%(dtype_name, dtype_name, dtype_name,
                                                 dtype_name, dtype_name))
        if not signature_only:
            code.append("""
    _verify_all_legacy_ndarrays("{op_name}", "{func_name}", ndargs, out)
            """.format(op_name=op_name, func_name=func_name))
            code.append("""
    return _imperative_invoke(%d, ndargs, keys, vals, out, False, %s)"""%(
        handle.value, str(output_is_list)))
    # return _imperative_invoke(%d, ndargs, keys, vals, out, %s)"""%(
    #     handle.value, str(output_is_list)))
        else:
            code.append("""
    return (0,)""")

    doc_str_lines = _os.linesep+''.join(['    '+s if s.strip() else s
                                         for s in 'r"""{doc_str}"""'.format(doc_str=doc_str)
                                         .splitlines(True)])
    code.insert(1, doc_str_lines)
    return ''.join(code), doc_str, is_np_op, kwarg_names, dtype_name


# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle, name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code, doc_str, is_np_op, kwargs_names, dtype = _generate_ndarray_function_code(handle, name, func_name)

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    # if func_name in rest_ops_set:
    #     print(func_name, kwargs_names, dtype)
    if func_name == "product":
        print(code)
    ndarray_function = local[func_name]
    ndarray_function.__name__ = func_name
    ndarray_function.__doc__ = doc_str
    ndarray_function.__module__ = 'mxnet.ndarray'
    return ndarray_function

_init_op_module('mxnet', 'ndarray', _make_ndarray_function)

# Update operator documentation with added float support
# Note that we can only do this after the op module is initialized
# Otherwise the backend operators cannot be found
# pylint: disable=wrong-import-position
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
adamw_update.__doc__ = _adamw_update.__doc__.replace("rescale_grad : NDArray",
                                                     "rescale_grad : NDArray or float")
mp_adamw_update.__doc__ = _mp_adamw_update.__doc__.replace("rescale_grad : NDArray",
                                                           "rescale_grad : NDArray or float")
