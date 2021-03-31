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

"""Namespace for the operators not belonging to the official numpy package
used in Gluon dispatched by F=ndarray module."""

import numpy as _np
from .. import numpy as np
from .._internal import NDArrayBase
from . import _api_internal
from ...util import set_module


__all__ = ['softmax', 'log_softmax', 'masked_softmax', 'masked_log_softmax', 'activation',
           'batch_norm', 'fully_connected', 'pick', 'convolution', 'deconvolution', 'pooling',
           'dropout', 'one_hot', 'rnn', 'embedding', 'topk', 'layer_norm', 'leaky_relu', 'sequence_mask',
           'batch_dot', 'broadcast_like', 'arange_like']



def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    if dtype and not isinstance(dtype, str):
        dtype = _np.dtype(dtype).name
    if use_length:
        assert length is not None, "Missing length input"
        return _api_internal.softmax(data, length, axis, temperature, True, dtype)
    else:
        assert length is None, "Length input is not used"
        return _api_internal.softmax(data, axis, temperature, False, dtype)

def log_softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    if dtype and not isinstance(dtype, str):
        dtype = _np.dtype(dtype).name
    if use_length:
        assert length is not None, "Missing length input"
        return _api_internal.log_softmax(data, length, axis, temperature, True, dtype)
    else:
        assert length is None, "Length input is not used"
        return _api_internal.log_softmax(data, axis, temperature, False, dtype)

def masked_softmax(data, mask, axis=-1, temperature=1.0, dtype=None):
    if mask is not None:
        neg = -1e18
        if _np.dtype(dtype) == _np.float16:
            neg = -1e4
        data = np.where(mask, data, neg)
        logits = (softmax(data, axis=axis) / temperature) * mask
    else:
        logits = softmax(data, axis=axis) / temperature
    return logits

def masked_log_softmax(data, mask, axis=-1, temperature=1.0, dtype=None):
    if mask is not None:
        neg = -1e18
        inf = -_np.inf
        if _np.dtype(dtype) == _np.float16:
            neg = -1e4
        data = np.where(mask, data, neg)
        logits = np.where(mask, log_softmax(data, axis=axis) / temperature, inf)
    else:
        logits = log_softmax(data, axis=axis) / temperature
    return logits

def activation(data, act_type='relu', name='fwd'):
    return _api_internal.activation(data, act_type)

def batch_norm(x, gamma, beta, running_mean, running_var, name='fwd', eps=1e-3, momentum=0.9,
               fix_gamma=True, use_global_stats=False, output_mean_var=False, axis=1,
               cudnn_off=False, min_calib_range=None, max_calib_range=None):
    return _api_internal.batch_norm(x, gamma, beta, running_mean, running_var, eps, momentum,
                                    fix_gamma, use_global_stats, output_mean_var, axis,
                                    cudnn_off, min_calib_range, max_calib_range)

def fully_connected(x, weight, bias=None, num_hidden=None, no_bias=True, flatten=True):
    assert num_hidden is not None, "Please provide number of hidden layers"
    if no_bias:
        return _api_internal.fully_connected(x, weight, num_hidden, no_bias, flatten)
    else:
        assert bias is not None, "Missing bias input"
        return _api_internal.fully_connected(x, weight, bias, num_hidden, no_bias, flatten)

def pick(data, index, axis=-1, mode='clip', keepdims=False):
    return _api_internal.pick(data, index, axis, mode, keepdims)

def convolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                pad=None, num_filter=1, num_group=1, workspace=1024, no_bias=False,
                cudnn_tune="off", cudnn_off=False, layout=None):
    assert data is not None and weight is not None, "Missing input data and weight"
    if no_bias:
        assert bias is None, "Using no bias"
        return _api_internal.convolution(data, weight, kernel, stride, dilate, pad,
                                         num_filter, num_group, workspace, no_bias,
                                         cudnn_tune, cudnn_off, layout)
    else:
        assert bias is not None, "Using bias"
        return _api_internal.convolution(data, weight, bias, kernel, stride, dilate, pad,
                                         num_filter, num_group, workspace, no_bias,
                                         cudnn_tune, cudnn_off, layout)

def deconvolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                  pad=None, adj=None, target_shape=None, num_filter=1, num_group=1,
                  workspace=512, no_bias=False, cudnn_tune="off",
                  cudnn_off=False, layout=None):
    assert data is not None and weight is not None, "Missing input data and weight"
    if no_bias:
        assert bias is None, "Using no bias"
        return _api_internal.deconvolution(data, weight, kernel, stride, dilate, pad,
                                           adj, target_shape, num_filter, num_group,
                                           workspace, no_bias, cudnn_tune, cudnn_off, layout)
    else:
        assert bias is not None, "Using bias"
        return _api_internal.deconvolution(data, weight, bias, kernel, stride, dilate, pad,
                                           adj, target_shape, num_filter, num_group,
                                           workspace, no_bias, cudnn_tune, cudnn_off, layout)

def pooling(data=None, kernel=None, stride=None, pad=None, pool_type="max",
            pooling_convention="valid", global_pool=False, cudnn_off=False,
            p_value=None, count_include_pad=None, layout=None, **kwargs):
    assert data is not None and kernel is not None, "Missing input data or kernel"
    out = _api_internal.pooling(data, kernel, stride, pad, pool_type, pooling_convention,
                                global_pool, cudnn_off, p_value, count_include_pad, layout)
    if isinstance(out, NDArrayBase):
        return out
    else:
        return list(out)

def dropout(data, p=0.5, mode="training", axes=None, cudnn_off=True, **kwargs):
    return _api_internal.dropout(data, p, mode, axes, cudnn_off)

def one_hot(data, depth=None, on_value=None, off_value=None, dtype=None):
    assert depth is not None, "Please provide the depth of one hot dimension."
    if dtype is not None and not isinstance(dtype, str):
            dtype = _np.dtype(dtype).name
    return _api_internal.one_hot(data, depth, on_value, off_value, dtype)

def rnn(data=None, parameters=None, state=None, state_cell=None, sequence_length=None,
        mode=None, state_size=None, num_layers=None, bidirectional=False,
        state_outputs=False, p=0.0, use_sequence_length=False, projection_size=None,
        lstm_state_clip_min=None, lstm_state_clip_max=None, lstm_state_clip_nan=None):
    assert mode is not None, "Please provide rnn type to compute. e.g. rnn_relu, rnn_tanh, lstm, gru"
    assert data is not None and parameters is not None and state is not None, \
        "Missing input data/parameters/state."
    assert state_size is not None, "Please provide state_size"
    assert num_layers is not None, "Please provide num_layers"
    if use_sequence_length:
        assert sequence_length is not None, \
            "use_sequence_length is set True, but no sequence_length provided."
        if mode=="lstm":
            assert state_cell is not None, \
                "RNN computing mode is lstm, but no state_cell is provided"
            return _api_internal.rnn(data, parameters, state, state_cell, sequence_length,
                                     state_size, num_layers, bidirectional, state_outputs,
                                     mode, p, use_sequence_length, projection_size,
                                     lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan)
        else:
            return _api_internal.rnn(data, parameters, state, sequence_length,
                                     state_size, num_layers, bidirectional, state_outputs,
                                     mode, p, use_sequence_length, projection_size,
                                     lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan)
    else:
        if mode=="lstm":
            assert state_cell is not None, \
                "RNN computing mode is lstm, but no state_cell is provided"
            return _api_internal.rnn(data, parameters, state, state_cell,
                                     state_size, num_layers, bidirectional, state_outputs,
                                     mode, p, use_sequence_length, projection_size,
                                     lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan)
        else:
            return _api_internal.rnn(data, parameters, state,
                                     state_size, num_layers, bidirectional, state_outputs,
                                     mode, p, use_sequence_length, projection_size,
                                     lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan)

def embedding(data, weight, input_dim=None, output_dim=None, dtype="float32", sparse_grad=False,
              **kwargs):
    assert input_dim > 1, "Vocabulary size of the input indices should be greater than 1."
    assert output_dim > 1, "Dimension of the embedding vectors should greater than 1."
    return _api_internal.embedding(data, weight, input_dim, output_dim, dtype, sparse_grad)

def topk(data, axis=-1, k=1, ret_typ="indices", is_ascend=False, dtype="float32"):
    out = _api_internal.topk(data, axis, k, ret_typ, is_ascend, dtype)
    if isinstance(out, NDArrayBase):
        return out
    return list(out)

def layer_norm(data=None, gamma=None, beta=None, axis=None, eps=None, output_mean_var=None):
    out = _api_internal.layer_norm(data, gamma, beta, axis, eps, output_mean_var)
    if isinstance(out, NDArrayBase):
        return out
    return list(out)

def leaky_relu(data=None, gamma=None, act_type="leaky", slope=None, lower_bound=None,
               upper_bound=None, **kwargs):
    if act_type == "prelu":
        assert gamma is not None, "If activation function is prelu, please provide input gamma"
        out = _api_internal.leaky_relu(data, gamma, act_type, slope, lower_bound, upper_bound)
        if isinstance(out, NDArrayBase):
            return out
        return list(out)
    else:
        return _api_internal.leaky_relu(data, act_type, slope, lower_bound, upper_bound)


def sequence_mask(data, sequence_length=None, use_sequence_length=False, value=0.0, axis=0):
    if use_sequence_length:
        assert sequence_length is not None, \
            "use_sequence_length flag is on, but no sequence_length provided"
        return _api_internal.sequence_mask(data, sequence_length, use_sequence_length,
                                           value, axis)
    else:
        return _api_internal.sequence_mask(data, False, value, axis)


def batch_dot(a, b, transpose_a=False, transpose_b=False, forward_stype="default"):
    return _api_internal.batch_dot(a, b, transpose_a, transpose_b, forward_stype)


def broadcast_like(lhs, rhs, lhs_axes=None, rhs_axes=None):
    return _api_internal.broadcast_like(lhs, rhs, lhs_axes, rhs_axes)

def arange_like(data, start=0.0, step=1.0, repeat=1, ctx=None, axis=None):
    return _api_internal.arange_like(data, start, step, repeat, ctx, axis)
