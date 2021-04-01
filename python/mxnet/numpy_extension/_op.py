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

"""Namespace for registering numpy_extension ops for imperative programming."""

from ..ndarray import numpy_extension as _mx_nd_npx
from ..util import set_module


__all__ = ['softmax', 'log_softmax', 'masked_softmax', 'masked_log_softmax', 'activation',
           'batch_norm', 'fully_connected', 'pick', 'convolution', 'deconvolution', 'pooling',
           'dropout', 'one_hot', 'rnn', 'embedding', 'topk', 'layer_norm', 'leaky_relu', 'sequence_mask',
           'batch_dot', 'broadcast_like', 'arange_like']

def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    r"""Applies the softmax function.

    The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

    .. math::
       softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}

    for :math:`j = 1, ..., K`

    t is the temperature parameter in softmax function. By default, t equals 1.0

    Parameters
    ----------
    data : NDArray
        The input array.

    length : NDArray
        The length array.

    axis : int, optional, default='-1'
        The axis along which to compute softmax.

    temperature : double or None, optional, default=None
        Temperature parameter in softmax

    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to the same as input's dtype if not defined (dtype=None).

    use_length : boolean or None, optional, default=0
        Whether to use the length input as a mask over the data input.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> data 
      x = [[ 1.  1.  1.]
           [ 1.  1.  1.]]

      softmax(x,axis=0) = [[ 0.5  0.5  0.5]
                           [ 0.5  0.5  0.5]]

      softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
                           [ 0.33333334,  0.33333334,  0.33333334]]
    """
    return _mx_nd_npx.softmax(data, length, axis=axis, temperature=temperature,
                              use_length=use_length, dtype=dtype)

def log_softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    r"""Computes the log softmax of the input.
    This is equivalent to computing softmax followed by log.

    Parameters
    ----------
    data : NDArray
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax
    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to the same as input's dtype if not defined (dtype=None).
    use_length : boolean or None, optional, default=0
        Whether to use the length input as a mask over the data input.
    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    >>> data = np.array([1, 2, .1])
    >>> npx.log_softmax(data)
    array([-1.4170278, -0.4170278, -2.3170278])
    >>> data = np.array([[1, 2, .1],[.1, 2, 1]])
    >>> npx.log_softmax(data, axis=0)
    array([[-0.34115386, -0.6931472 , -1.2411538 ],
        [-1.2411538 , -0.6931472 , -0.34115386]])
    """
    return _mx_nd_npx.log_softmax(data, length, axis=axis, temperature=temperature,
                                  use_length=use_length, dtype=dtype)

def masked_softmax(data, mask, axis=-1, temperature=1.0, normalize=True):
    return _mx_nd_npx.masked_softmax(data, mask, axis=axis, temperature=temperature,
                                     normalize=normalize)

def masked_log_softmax(data, mask, axis=-1, temperature=1.0, normalize=True):
    return _mx_nd_npx.masked_log_softmax(data, mask, axis=axis, temperature=temperature,
                                         normalize=normalize)

def activation(data, act_type='relu', name='fwd'):
    return _mx_nd_npx.activation(data, act_type=act_type)

def batch_norm(x, gamma, beta, running_mean, running_var, name='fwd', eps=1e-3, momentum=0.9,
               fix_gamma=True, use_global_stats=False, output_mean_var=False, axis=1,
               cudnn_off=False, min_calib_range=None, max_calib_range=None):
    return _mx_nd_npx.batch_norm(x, gamma, beta, running_mean, running_var, eps=eps,
                                 momentum=momentum, fix_gamma=fix_gamma,
                                 use_global_stats=use_global_stats,
                                 output_mean_var=output_mean_var, axis=axis, cudnn_off=cudnn_off,
                                 min_calib_range=min_calib_range, max_calib_range=max_calib_range)

def fully_connected(x, weight, bias=None, num_hidden=None, no_bias=True, flatten=True):
    return _mx_nd_npx.fully_connected(x, weight, bias=bias, num_hidden=num_hidden,
                                      no_bias=no_bias, flatten=flatten)

def pick(data, index, axis=-1, mode='clip', keepdims=False):
    return _mx_nd_npx.pick(data, index, axis, mode, keepdims)


def convolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                pad=None, num_filter=1, num_group=1, workspace=1024, no_bias=False,
                cudnn_tune="off", cudnn_off=False, layout=None):
    return _mx_nd_npx.deconvolution(data=data, weight=weight, bias=bias, kernel=kernel,
                                    stride=stride, dilate=dilate, pad=pad,num_filter=num_filter,
                                    num_group=num_group, workspace=workspace, no_bias=no_bias,
                                    cudnn_tune=cudnn_tune, cudnn_off=cudnn_off, layout=layout)

def deconvolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                  pad=None, adj=None, target_shape=None, num_filter=1, num_group=1,
                  workspace=512, no_bias=False, cudnn_tune="off",
                  cudnn_off=False, layout=None):
    return _mx_nd_npx.deconvolution(data=data, weight=weight, bias=bias, kernel=kernel,
                                    stride=stride, dilate=dilate, pad=pad, adj=adj,
                                    target_shape=target_shape, num_filter=num_filter,
                                    num_group=num_group, workspace=workspace, no_bias=no_bias,
                                    cudnn_tune=cudnn_tune, cudnn_off=cudnn_off, layout=layout)


def pooling(data=None, kernel=None, stride=None, pad=None, pool_type="max",
            pooling_convention="valid", global_pool=False, cudnn_off=False,
            p_value=None, count_include_pad=None, layout=None, **kwargs):
    return _mx_nd_npx.pooling(data=data, kernel=kernel, stride=stride, pad=pad,
                              pool_type=pool_type, pooling_convention=pooling_convention,
                              global_pool=global_pool, cudnn_off=cudnn_off, p_value=p_value,
                              count_include_pad=count_include_pad, layout=layout)


def dropout(data, p=0.5, mode="training", axes=None, cudnn_off=True, **kwargs):
    return _mx_nd_npx.dropout(data=data, p=p, mode=mode, axes=axes, cudnn_off=cudnn_off)

def one_hot(data, depth=None, on_value=None, off_value=None, dtype=None):
    return _mx_nd_npx.one_hot(data=data, depth=depth, on_value=on_value, off_value=off_value,
                              dtype=dtype)

def rnn(data=None, parameters=None, state=None, state_cell=None, sequence_length=None,
        mode=None, state_size=None, num_layers=None, bidirectional=False,
        state_outputs=False, p=0.0, use_sequence_length=False, projection_size=None,
        lstm_state_clip_min=None, lstm_state_clip_max=None, lstm_state_clip_nan=None):
    return _mx_nd_npx.rnn(data=data, parameters=parameters, state=state, state_cell=state_cell,
                          sequence_length=sequence_length, mode=mode, state_size=state_size,
                          num_layers=num_layers, bidirectional=bidirectional,
                          state_outputs=state_outputs, p=p, use_sequence_length=use_sequence_length,
                          projection_size=projection_size, lstm_state_clip_min=lstm_state_clip_min,
                          lstm_state_clip_max=lstm_state_clip_max,
                          lstm_state_clip_nan=lstm_state_clip_nan)


def embedding(data, weight, input_dim=None, output_dim=None, dtype="float32", sparse_grad=False,
              **kwargs):
    return _mx_nd_npx.embedding(data=data, weight=weight, input_dim=input_dim, output_dim=output_dim,
                                dtype=dtype, sparse_grad=sparse_grad)


def topk(data, axis=-1, k=1, ret_typ="indices", is_ascend=False, dtype="float32"):
    return _mx_nd_npx.topk(data=data, axis=axis, k=k, ret_typ=ret_typ, is_ascend=is_ascend, dtype=dtype)

def layer_norm(data=None, gamma=None, beta=None, axis=None, eps=None, output_mean_var=None):
    return _mx_nd_npx.layer_norm(data=data, gamma=gamma, beta=beta, axis=axis, eps=eps,
                                 output_mean_var=output_mean_var)

def leaky_relu(data=None, gamma=None, act_type="leaky", slope=None, lower_bound=None,
               upper_bound=None, **kwargs):
    return _mx_nd_npx.leaky_relu(data=data, gamma=gamma, act_type=act_type, slope=slope,
                                 lower_bound=lower_bound, upper_bound=upper_bound)

def sequence_mask(data, sequence_length=None, use_sequence_length=False, value=0.0, axis=0):
    return _mx_nd_npx.sequence_mask(data=data, sequence_length=sequence_length,
                                    use_sequence_length=use_sequence_length, value=value,
                                    axis=axis)

def batch_dot(a, b, transpose_a=False, transpose_b=False, forward_stype="default"):
    return _mx_nd_npx.batch_dot(a=a, b=b, transpose_a=transpose_a,
                                transpose_b=transpose_b, forward_stype=forward_stype)

def broadcast_like(lhs, rhs, lhs_axes=None, rhs_axes=None):
    return _mx_nd_npx.broadcast_like(lhs=lhs, rhs=rhs, lhs_axes=lhs_axes, rhs_axes=rhs_axes)

def arange_like(data, start=0.0, step=1.0, repeat=1, ctx=None, axis=None):
    return _mx_nd_npx.arange_like(data=data, start=start, step=step, repeat=repeat,
                                  ctx=ctx, axis=axis)
