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

"""Namespace for registering control flow ops for imperative programming."""

import copy
import ctypes

from ..util import set_module
from ..numpy import ndarray as np_ndarray
from ..symbol import Symbol
from ..base import check_call, _LIB, _as_list, SymbolHandle
from .. import numpy as _mx_np, symbol, _deferred_compute as dc, autograd as ag
from ..attribute import AttrScope, current as current_attribute
from ..ndarray import numpy_extension as _mx_nd_npx


__all__ = ["foreach", "while_loop", "cond"]


def _flatten(args, inout_str):
    """Parse the arguments into a flattened list + an additional format array.
    The format array stores the structure of the original arguments to help reconstruct the inputs.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    inout_str : str
        The name of the HybridBlock

    Returns
    -------
    flat : list of Symbol or NDArray
        The flatten version of the input args.
    fmts : (nested) list of ints
        Stores the format information of the original structured args.
    """
    if isinstance(args, np_ndarray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)
    if args is None:
        return [None], int(-1)

    if not isinstance(args, (list, tuple)):
        raise ValueError("When hybridized, the input of HybridBlock {}"
                         " must be (nested) list of Symbol"
                         " or NDArray, "
                         "but got {} of type {}".format(inout_str, str(args), str(type(args))))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i, inout_str)
        flat.extend(arg)
        fmts.append(fmt)
    return flat, fmts


def _regroup(args, fmt):
    """Reconstruct the structured arguments based on the flattened version.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    fmt : (nested) list of ints
        Stores the format information of the original structured args.

    Returns
    -------
    ret : NDArray, Symbol, or (nested) list of Symbol or NDArray

    """
    def _merger(args, fmt):
        """Recursive call to merge the arguments"""
        if isinstance(fmt, int):
            if fmt < -1:
                raise ValueError("Unsupported encoded format {}.".format(fmt))
            if fmt == 0:
                return args[0], args[1:]
            if fmt == -1:
                if args[0] is not None:
                    raise ValueError('We do not support passing types that are not None'
                                     ' when the initial HybridBlock has received NoneType and'
                                     ' has been hybridized.'
                                     ' Received arg = {}, fmt = {}.'.format(args[0], fmt))
                return None, args[1:]
            else:
                return args[:fmt], args[fmt:]

        if not isinstance(args, (list, tuple)):
            raise ValueError("When hybridized, the output of HybridBlock must be (nested)"
                             " list of Symbol or NDArray, "
                             "but got {} of type {}".format(args, type(args)))
        ret = []
        for i in fmt:
            res, args = _merger(args, i)
            ret.append(res)
        return ret, args
    return _merger(args, fmt)[0]

# We want to generate a unique name for input symbols to a control flow
# operator. The names are generated on purpose differently from the symbols
# cut from the graph.
def _get_sym_uniq_name(sym):
    return '{}-{}'.format(sym.name, sym.attr('_value_index'))

def _cut_subgraph(subg):
    num_handles = ctypes.c_int(0)
    handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolCutSubgraph(subg.handle, ctypes.byref(handles),
                                        ctypes.byref(num_handles)))

    syms = []
    for i in range(num_handles.value):
        s = Symbol(ctypes.cast(handles[i], SymbolHandle))
        syms.append(s)
    return syms

def _get_unique_subgraph_name(subgraph_name):
    attrs = current_attribute()._attr
    if attrs.get("__subgraph_name__", "") != "":
        subgraph_name = "".join([attrs["__subgraph_name__"], "$", subgraph_name])
    AttrScope._subgraph_names[subgraph_name] += 1
    subgraph_name = subgraph_name + str(AttrScope._subgraph_names[subgraph_name] - 1)
    return subgraph_name

# This construct a subgraph for given output nodes.
# If an output node is one of the input nodes, we call identity to make sure
# that outputs nodes are different from input nodes.
def _construct_subgraph(sym_out, sym_states, name):
    sym_out = _as_list(sym_out)
    sym_states = _as_list(sym_states)
    all_outputs = []
    all_outputs.extend(sym_out)
    all_outputs.extend(sym_states)
    g = symbol.Group(all_outputs)

    flat_out = []
    all_input_names = g.list_inputs()
    output_names = {o.name for o in sym_out}
    for o in sym_out:
        if o.name in all_input_names or o.list_attr().get("__subgraph_name__", "") != name:
            flat_out.append(symbol.op.identity(o))
        else:
            flat_out.append(o)

    for s in sym_states:
        if s.name in all_input_names or s.name in output_names or \
           s.list_attr().get("__subgraph_name__", "") != name:
            flat_out.append(symbol.op.identity(s))
        else:
            flat_out.append(s)
    return symbol.Group(flat_out)

@set_module('mxnet.numpy_extension')
def foreach(body, data, init_states, name="foreach"):
    """Run a for loop with user-defined computation over NDArrays on dimension 0.

    This operator simulates a for loop and body has the computation for an iteration
    of the for loop. It runs the computation in body on each slice from the input
    NDArrays.

    body takes two arguments as input and outputs a tuple of two elements,
    as illustrated below::

        out, states = body(data1, states)

    data1 can be either an NDArray or a list of NDArrays. If data is an NDArray,
    data1 is an NDArray. Otherwise, data1 is a list of NDArrays and has the same
    size as data. states is a list of NDArrays and have the same size as init_states.
    Similarly, out can be either an NDArray or a list of NDArrays, which are concatenated
    as the first output of foreach; states from the last execution of body
    are the second output of foreach.

    The computation done by this operator is equivalent to the pseudo code below
    when the input data is NDArray::

        states = init_states
        outs = []
        for i in data.shape[0]:
            s = data[i]
            out, states = body(s, states)
            outs.append(out)
        outs = stack(*outs)


    Parameters
    ----------
    body : a Python function.
        Define computation in an iteration.
    data: an NDArray or a list of NDArrays.
        The input data.
    init_states: an NDArray or nested lists of NDArrays.
        The initial values of the loop states.

    Returns
    -------
    outputs: an NDArray or nested lists of NDArrays.
        The output data concatenated from the output of all iterations.
    states: an NDArray or nested lists of NDArrays.
        The loop states in the last iteration.

    Examples
    --------
    >>> step = lambda data, states: (data + states[0], [states[0] * 2])
    >>> data = mx.np.random.uniform(size=(2, 10))
    >>> states = [mx.np.random.uniform(size=(10))]
    >>> outs, states = npx.control_flow.foreach(step, data, states)
    """

    def check_input(inputs, in_type, msg):
        is_NDArray_or_list = True
        if isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, in_type):
                    is_NDArray_or_list = False
                    break
        else:
            is_NDArray_or_list = isinstance(inputs, in_type)
        assert is_NDArray_or_list, msg

    flatten_data, data_fmt = _flatten(data, "foreach input")
    check_input(flatten_data, np_ndarray,
                "data should be an mxnet.numpy.ndarray or a nested list of mxnet.numpy.ndarray")
    flatten_state, state_fmt = _flatten(init_states, "foreach states")
    check_input(flatten_state, np_ndarray,
                "init_states should be an mxnet.numpy.ndarray or a nested list of mxnet.numpy.ndarray")

    real_data = [ele.detach() if ele is not None else None for ele in flatten_data]
    real_state = [ele.detach() if ele is not None else None for ele in flatten_state]

    # If the input python function references to the symbols outside
    # the python function, we need to prune the computation graph constructed from
    # the function. One way of doing it is to mark the nodes in the computation graph
    # with AttrScope and prune the nodes without the special attribute.
    name = _get_unique_subgraph_name(name)
    with AttrScope(__subgraph_name__=name):
        data_names = ['data_subgraph{}'.format(i) for i, ele in enumerate(real_data)]
        state_names = ['state_subgraph{}'.format(i) for i, ele in enumerate(real_state)]
        symbol_data = [
            symbol.var(name).as_np_ndarray()
            for arg, name in zip(real_data, data_names)
        ]
        symbol_state = [
            symbol.var(name).as_np_ndarray()
            for arg, name in zip(real_state, state_names)
        ]
        dc.set_variable(real_data, symbol_data)
        dc.set_variable(real_state, symbol_state)
        in_eles = _regroup(real_data, data_fmt)
        in_states = _regroup(real_state, state_fmt)
        if dc.is_deferred_compute():
            out, states = body(in_eles, in_states)
        else:
            with ag.pause(), dc.context():
                out, states = body(in_eles, in_states)

        flatten_out, out_fmt = _flatten(out, "foreach output")
        flatten_out_state, state_fmt = _flatten(states, "foreach loop_vars")

        num_out_data = len(flatten_out)
        num_states = len(flatten_out_state)
        num_outputs = num_out_data + num_states
        sym_out = dc.get_symbol(flatten_out)
        sym_states = dc.get_symbol(flatten_out_state)
        g = _construct_subgraph(sym_out, sym_states, name)


    subg_input_names = g.list_inputs()

    ordered_ins = [x for x in flatten_data]

    in_data_locs = []
    for dname in data_names:
        # Some data may not be used.
        if dname in subg_input_names:
            in_data_locs.append(subg_input_names.index(dname))
        else:
            raise AssertionError("the data arrays have to be used in the loop body")

    ordered_ins.extend(flatten_state)
    # this defines the location of state_syms in the list of subgraph inputs.
    in_state_locs = []
    for sname in state_names:
        # Some state may not be used.
        if sname in subg_input_names:
            in_state_locs.append(subg_input_names.index(sname))
        else:
            raise AssertionError("the state arrays have to be used in the loop body")

    ret = _mx_nd_npx.foreach(g, *ordered_ins, num_outputs=num_outputs,
                             num_out_data=num_out_data, in_state_locs=in_state_locs,
                             in_data_locs=in_data_locs, remain_locs=[])
    outs = []
    for i in range(num_outputs - num_states):
        outs.append(ret[i])
    outs, _ = _regroup(outs, out_fmt)
    states = []
    for i in range(num_states):
        states.append(ret[num_outputs - num_states + i])
    states, _ = _regroup(states, state_fmt)

    return (outs, states)


#pylint: disable=W0621
@set_module('mxnet.numpy_extension')
def while_loop(cond, func, loop_vars, max_iterations=None):
    """Run a while loop with user-defined computation and loop condition.

    This operator simulates a while loop which iterately does customized computation
    as long as the condition is satisfied.

    `loop_vars` is a list of NDArrays on which the computation uses.

    `cond` is a user-defined function, used as the loop condition.
    It consumes `loop_vars`, and produces a scalar MXNet NDArray,
    indicating the termination of the loop.
    The loop ends when `cond` returns false (zero).
    The `cond` is variadic, and its signature should be
    `cond(*loop_vars) => NDArray`.

    `func` is a user-defined function, used as the loop body.
    It also consumes `loop_vars`, and produces `step_output` and `new_loop_vars` at each step.
    In each step, `step_output` should contain the same number elements.
    Through all steps, the i-th element of `step_output` should have the same shape and dtype.
    Also, `new_loop_vars` should contain the same number of elements as `loop_vars`,
    and the corresponding element should have the same shape and dtype.
    The `func` is variadic, and its signature should be
    `func(*loop_vars) =>
    (NDArray or nested List[NDArray] step_output, NDArray or nested List[NDArray] new_loop_vars)`.

    `max_iterations` is a scalar that defines the maximum number of iterations allowed.

    This function returns two lists.
    The first list has the length of `|step_output|`,
    in which the i-th element are all i-th elements of
    `step_output` from all steps, stacked along axis 0.
    The second list has the length of `|loop_vars|`,
    which represents final states of loop variables.

    .. warning::

       For now, the axis 0 of all NDArrays in the first list are `max_iterations`,
       due to lack of dynamic shape inference.

    .. warning::

       When `cond` is never satisfied, we assume `step_output` is empty,
       because it cannot be inferred. This is different from the symbolic version.

    Parameters
    ----------
    cond: a Python function.
        The loop condition.
    func: a Python function.
        The loop body.
    loop_vars: an NDArray or nested lists of NDArrays.
        The initial values of the loop variables.
    max_iterations: a python int.
        Maximum number of iterations.

    Returns
    ------
    outputs: an NDArray or nested lists of NDArrays
        stacked output from each step
    states: an NDArray or nested lists of NDArrays
        final state

    Examples
    --------
    >>> cond = lambda i, s: i <= 5
    >>> func = lambda i, s: ([i + s], [i + 1, s + i])
    >>> loop_vars = (mx.np.array([0], dtype="int64"), mx.np.array([1], dtype="int64"))
    >>> outputs, states = mx.npx.while_loop(cond, func, loop_vars, max_iterations=10)
    >>> outputs
    [
    [[ 1]
    [ 2]
    [ 4]
    [ 7]
    [11]
    [16]
    [...]  # undefined value
    [...]
    [...]
    [...]]
    <NDArray 6x1 @cpu(0)>]
    >>> states
    [
    [6]
    <NDArray 1 @cpu(0)>,
    [16]
    <NDArray 1 @cpu(0)>]
    """
    def _to_python_scalar(inputs, type_, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if isinstance(inputs, np_ndarray):
            inputs = inputs.item()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError("Cannot convert %s to python %s" % (name, type_.__name__))
        return inputs

    def _func_wrapper(loop_vars):
        """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (None or tuple of step_outputs, tuple of new_loop_vars)
        """
        step_output, new_loop_vars = func(*loop_vars)
        if step_output is None:
            step_output = []
        if new_loop_vars is None:
            new_loop_vars = []
        if isinstance(step_output, tuple):
            step_output = list(step_output)
        if isinstance(new_loop_vars, tuple):
            new_loop_vars = list(new_loop_vars)
        new_loop_vars = _as_list(new_loop_vars)
        if len(loop_vars) != len(new_loop_vars):
            raise ValueError("The length of loop_vars should be consistent during the loop")
        return step_output, new_loop_vars

    if max_iterations is None:
        raise ValueError("max_iterations should be specified")
    max_iterations = _to_python_scalar(max_iterations, int, "max_iteration")
    # It should be work as fine if loop_vars are empty I guess,
    # but it is semantically unnecessary to include this case.
    if len(loop_vars) == 0:
        raise ValueError("loop_vars should contain at least one element")

    steps = 0
    outputs = []
    # there might not be an iteration.
    out_fmt = None
    not_loop_var_list = isinstance(loop_vars, np_ndarray)
    loop_vars = _as_list(loop_vars)
    while steps < max_iterations and \
            _to_python_scalar(cond(*loop_vars), bool, "Return value of cond"): # loop condition
        step_output, loop_vars = _func_wrapper(loop_vars)
        step_output, out_fmt = _flatten(step_output, "while output")
        outputs.append(step_output)
        steps += 1
        if len(outputs) != steps or len(step_output) != len(outputs[0]):
            raise ValueError("Number of elements in step_output should be the same in each step")
    stacked_outputs = []
    for i_th, items in enumerate(zip(*outputs), 1):
        # `mx.ndarray.pad` only support 4-D or 5-D inputs for now
        # so we could not use it.
        items = [_mx_np.expand_dims(x, 0) for x in items]
        try:
            concate_outputs = _mx_np.concatenate(items, axis=0)
            print(concate_outputs.shape)
            if steps != max_iterations and items:
                to_pad = max_iterations - steps
                concate_outputs = _mx_np.pad(concate_outputs, pad_width=((0, to_pad), (0, 0)))
            stacked_outputs.append(concate_outputs)
        except ValueError:
            raise ValueError("\n".join(
                ["Shapes of %d-th elements in step_outputs are inconsistent, which are:" % i_th] +
                ["  Step %d, shape is %s" % (i, str(x.shape)) for i, x in enumerate(items)]
            ))
    if out_fmt is not None:
        stacked_outputs, _ = _regroup(stacked_outputs, out_fmt)
    if not_loop_var_list:
        loop_vars = loop_vars[0]
    return stacked_outputs, loop_vars


@set_module('mxnet.numpy_extension')
def cond(pred, then_func, else_func):
    """Run an if-then-else using user-defined condition and computation

    This operator simulates a if-like branch which chooses to do one of
    the two customized computations according to the specified condition.

    `pred` is a scalar MXNet NDArray,
    indicating which branch of computation should be used.

    `then_func` is a user-defined function, used as computation of the then branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `then_func` should be
    `then_func() => NDArray or nested List[NDArray]`.

    `else_func` is a user-defined function, used as computation of the else branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `else_func` should be
    `else_func() => NDArray or nested List[NDArray]`.

    The `outputs` produces by `then_func` and `else_func` should have the same number
    of elements, all of which should be in the same shape, of the same dtype and stype.

    This function returns a list of symbols, representing the computation result.

    Parameters
    ----------
    pred: a MXNet NDArray representing a scalar.
        The branch condition.
    then_func: a Python function.
        The computation to be executed if `pred` is true.
    else_func: a Python function.
        The computation to be executed if `pred` is false.

    Returns
    -------
    outputs: an NDArray or nested lists of NDArrays, representing the result of computation.

    Examples
    --------
    >>> a, b = mx.nd.array([1]), mx.nd.array([2])
    >>> pred = a * b < 5
    >>> then_func = lambda: (a + 5) * (b + 5)
    >>> else_func = lambda: (a - 5) * (b - 5)
    >>> outputs = mx.nd.contrib.cond(pred, then_func, else_func)
    >>> outputs[0]
    [42.]
    """

    def _create_subgraph(graph_vars, graph_func, subgraph_name):
        subgraph_name = _get_unique_subgraph_name(subgraph_name)
        with AttrScope(__subgraph_name__=subgraph_name):
            # create new variables with the same name,
            # them feed them to the given func
            new_graph_vars = [symbol.var(sym.name) for sym in graph_vars]
            outputs = graph_func(*new_graph_vars)
            outputs, out_fmt = _flatten(outputs, "cond outputs")
            num_outputs = len(outputs)
            # nnvm cut-graph does not allow inputs and outputs overlap
            # so we calculate the name of inputs, and copy outputs once it overlaps with inputs
            # group all outputs of graph_func
            all_input_names = symbol.Group(outputs).list_inputs()
            in_input = lambda x: x.name in all_input_names
            in_graph = lambda x: x.list_attr().get("__subgraph_name__", "") == subgraph_name
            make_identity = lambda x: symbol.op.identity(x) if in_input(x) or not in_graph(x) \
                                      else x
            graph = symbol.Group(list(map(make_identity, outputs)))
        return graph, num_outputs, out_fmt

    def _union_inputs(*graphs):
        # Given a list of graphs, each whose inputs are either from input_vars or other variables.
        # 1) calculate a list `inputs`, the union of their inputs.
        # 2) for each graph, determine in which indices their inputs reside in `inputs`
        # 3) for each variable in the input of `graph`, find which index it is
        inputs = []             # List[Symbol], result of 1)
        locs = []               # List[Tuple(List[Int], List[Int])], a list of tuples,
                                # where tuples are results of 2) and 3)
        input_id_to_loc = {}    # Dict[int, int], given id(sym), input_id_to_loc maps it
                                # to a `loc`, where inputs[loc] = sym
        for graph in graphs:
            # some input_vars are inputs to `graph`, some are not
            name_to_input_vars = {sym.name: sym for sym in inputs}
            # other inputs to `graph` created by cut_graph
            name_to_cut_g_syms = {sym.list_outputs()[0]: sym for sym in _cut_subgraph(graph)}
            # input_syms: all inputs to the `graph`
            name_to_input_syms = {sym.name: sym for sym in _get_graph_inputs(graph)}
            # collect arguments for each subgraph
            input_locs = []                         # results from the second step
            for name in graph.list_inputs():
                assert name in name_to_input_syms   # it should obviously hold
                # name -> sym
                if name in name_to_input_vars:
                    sym = name_to_input_vars[name]
                elif name in name_to_cut_g_syms:
                    sym = name_to_cut_g_syms[name]
                else:
                    sym = copy.deepcopy(name_to_input_syms[name])
                # do 2), and 1) is implicitly done
                if id(sym) in input_id_to_loc:
                    loc = input_id_to_loc[id(sym)]
                else:
                    loc = len(input_id_to_loc)
                    inputs.append(sym)
                    input_id_to_loc[id(sym)] = loc
                input_locs.append(loc)
            locs.append(input_locs)
        return inputs, locs
    inputs = []
    # create graph for `cond_func'
    cond_g, cond_num_outputs, _ = _create_subgraph(inputs, lambda: pred, name + "_pred")
    if cond_num_outputs != 1:
        raise ValueError("pred should always be a single output")
    # create graph for `then`
    then_g, then_num_outputs, then_fmt = _create_subgraph(inputs, then_func, name + "_then")
    # create graph for `else`
    else_g, else_num_outputs, _ = _create_subgraph(inputs, else_func, name + "_else")
    if then_num_outputs != else_num_outputs:
        raise ValueError("Number of outputs differs between then-branch and else-branch")
    # find symbols used in either cond_g or func_g
    input_syms, (cond_input_locs, then_input_locs, else_input_locs) = \
        _union_inputs(cond_g, then_g, else_g)
    result = symbol._internal._cond(
        # [cond, then_g, else_g, *input_syms]
        cond_g,
        then_g,
        else_g,
        *input_syms,
        cond_input_locs=cond_input_locs,
        then_input_locs=then_input_locs,
        else_input_locs=else_input_locs,
        num_outputs=then_num_outputs
    )
    outputs = [result[i] for i in range(then_num_outputs)]
    outputs, _ = _regroup(outputs, then_fmt)
    return outputs
