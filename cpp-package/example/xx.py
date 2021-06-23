from collections import OrderedDict, defaultdict
from mxnet.gluon import Parameter
from mxnet import numpy_extension as _mx_npx
from mxnet.util import is_np_array, np_shape, np_array
import warnings
import re
import json
import weakref
import numpy as np

class Block:
    """Base class for all neural network layers and models. Your models should
    subclass this class.

    :py:class:`Block` can be nested recursively in a tree structure. You can create and
    assign child :py:class:`Block` as regular attributes::

        import mxnet as mx
        from mxnet.gluon import Block, nn

        class Model(Block):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                self.dense0 = nn.Dense(20)
                self.dense1 = nn.Dense(20)

            def forward(self, x):
                x = mx.nd.relu(self.dense0(x))
                return mx.nd.relu(self.dense1(x))

        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model(mx.nd.zeros((10, 10), ctx=mx.cpu(0)))


    Child :py:class:`Block` assigned this way will be registered and :py:meth:`collect_params`
    will collect their Parameters recursively. You can also manually register
    child blocks with :py:meth:`register_child`.

    """
    def __init__(self):
        self._children = OrderedDict()
        self._reg_params = {}
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()

    def __setattr__(self, name, value):
        """Registers parameters."""

        if hasattr(self, name):
            existing = getattr(self, name)
            if isinstance(existing, (Parameter, Block)) and not isinstance(value, type(existing)):
                raise TypeError('Changing attribute type for {name} from {type1} to {type2}' \
                                'is not allowed.'.format(
                                    name=name, type1=type(existing), type2=type(value)))

        if isinstance(value, Block):
            self.register_child(value, name)
        elif isinstance(value, Parameter):
            self._reg_params[name] = value

        super(Block, self).__setattr__(name, value)

    def _check_container_with_block(self):
        children = set(self._children.values())
        def _find_unregistered_block_in_container(data):
            # Find whether a nested container structure contains Blocks
            if isinstance(data, (list, tuple)):
                for ele in data:
                    if _find_unregistered_block_in_container(ele):
                        return True
                return False
            elif isinstance(data, dict):
                for _, v in data.items():
                    if _find_unregistered_block_in_container(v):
                        return True
                return False
            elif isinstance(data, Block):
                return not data in (c() for c in children)
            else:
                return False
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple, dict)) and not (k.startswith('__') or k == '_children'):
                if _find_unregistered_block_in_container(v):
                    warnings.warn('"{name}" is an unregistered container with Blocks. '
                                  'Note that Blocks inside the list, tuple or dict will not be '
                                  'registered automatically. Make sure to register them using '
                                  'register_child() or switching to '
                                  'nn.Sequential/nn.HybridSequential instead. '
                                  .format(name=self.__class__.__name__ + "." + k), stacklevel=3)

    def _alias(self):
        return self.__class__.__name__.lower()

    @property
    def params(self):
        """Returns this :py:class:`Block`'s parameter dictionary (does not include its
        children's parameters)."""
        return self._reg_params

    def collect_params(self, select=None):
        """Returns a :py:class:`Dict` containing this :py:class:`Block` and all of its
        children's Parameters(default), also can returns the select :py:class:`Dict`
        which match some given regular expressions.

        For example, collect the specified parameters in ['conv1.weight', 'conv1.bias', 'fc.weight',
        'fc.bias']::

            model.collect_params('conv1.weight|conv1.bias|fc.weight|fc.bias')

        or collect all parameters whose names end with 'weight' or 'bias', this can be done
        using regular expressions::

            model.collect_params('.*weight|.*bias')

        Parameters
        ----------
        select : str
            regular expressions

        Returns
        -------
        The selected :py:class:`Dict`
        """
        # We need to check here because blocks inside containers are not supported.
        self._check_container_with_block()
        return self._collect_params_with_prefix(select=select)

    def _collect_params_with_prefix(self, prefix='', select=None):
        if prefix:
            prefix += '.'
        if select is None:
            ret = {prefix + key : val for key, val in self._reg_params.items()}
        else:
            pattern = re.compile(select)
            ret = {prefix + key : val for key, val in self._reg_params.items() if pattern.match(prefix + key)}

        for name, child in self._children.items():
            ret.update(child()._collect_params_with_prefix(prefix + name, select))
        return ret

    def save_parameters(self, filename, deduplicate=False):
        """Save parameters to file.

        Saved parameters can only be loaded with `load_parameters`. Note that this
        method only saves parameters, not model structure. If you want to save
        model structures, please use :py:meth:`HybridBlock.export`.

        Parameters
        ----------
        filename : str
            Path to file.
        deduplicate : bool, default False
            If True, save shared parameters only once. Otherwise, if a Block
            contains multiple sub-blocks that share parameters, each of the
            shared parameters will be separately saved for every sub-block.

        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>`_
        """
        params = self._collect_params_with_prefix()

        if deduplicate:
            # Shared parameters are stored only a single time as of MXNet 1.6.
            # Shared parameters are registered under multiple prefixes returned by
            # _collect_params_with_prefix. We select a single one and only store
            # it. In load_parameters it is sufficient for a shared parameter to
            # only set it for a single prefix.
            reverse_params = {v: k for k, v in params.items()}
            params = {v: k for k, v in reverse_params.items()}

        arg_dict = {key: val._reduce() for key, val in params.items()}
        if is_np_array():
            _mx_npx.savez(filename, **arg_dict)
        else:
            ndarray.save(filename, arg_dict)

    def load_parameters(self, filename, ctx=None, allow_missing=False,
                        ignore_extra=False, cast_dtype=False, dtype_source='current'):
        """Load parameters from file previously saved by `save_parameters`.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any.
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>`_
        """
        if is_np_array():
            # failure may happen when loading parameters saved as NDArrays within
            # NumPy semantics. Check the failure type and recover from it if it happens.
            try:
                loaded = _mx_npx.load(filename)
            except MXNetError as e:
                err_msg = str(e)
                if 'is_np_shape' in err_msg:
                    # Loading failure due to parameters saved without numpy semantics.
                    # Temporarily disable numpy semantics and load parameters. After it's
                    # done, resume the numpy semantics. This is fine because the cases
                    # numpy ndarray covers is a superset of the legacy ndarray's.
                    with np_array(False):
                        with np_shape(False):
                            loaded_nds = ndarray.load(filename)
                    assert isinstance(loaded_nds, dict),\
                        'expecting a dict type, got {}'.format(str(type(loaded_nds)))
                    loaded = {k: loaded_nds[k].as_np_ndarray() for k in loaded_nds}
                else:
                    raise ValueError(err_msg)
        else:
            loaded = ndarray.load(filename)

        if not loaded:
            return
        full_dict = {'params': loaded, 'filename': filename}
        self.load_dict(full_dict, ctx, allow_missing, ignore_extra, cast_dtype, dtype_source)

    def load_dict(self, param_dict, ctx=None, allow_missing=False,
                  ignore_extra=False, cast_dtype=False, dtype_source="current"):
        """Load parameters from dict

        Parameters
        ----------
        param_dict : dict
            Dictionary containing model parameters
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represented in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this dict.
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        """
        if isinstance(param_dict.get('filename'), str):
            # pass from load_parameters
            filename = param_dict['filename']
            param_dict = param_dict['params']
        else:
            filename = None
        params = self.collect_params()
        error_str = "file: %s" % (filename) if filename else "param_dict"
        loaded = {k[4:] if k.startswith('arg:') or k.startswith('aux:') else k: v \
                  for k, v in param_dict.items()}

        if not allow_missing:
            params_inv = defaultdict(list)
            for k, v in params.items():
                params_inv[v].append(k)

            for name, param in params.items():
                assert any(p in loaded for p in params_inv[param]), \
                    "Parameter '%s' is missing in '%s', which contains parameters: %s. " \
                    "Set allow_missing=True to ignore missing parameters."%(
                        name, error_str, _brief_print_list(loaded.keys()))

        if ctx is None:
            ctx = _context.current_context()
        for name in loaded:
            if not ignore_extra and name not in params:
                raise ValueError(
                    "Parameter '%s' loaded from '%s' is not present in Dict, " \
                    "which contains parameters %s. Set ignore_extra=True to ignore. "%(
                        name, error_str, _brief_print_list(params.keys())))
            if name in params:
                param = loaded[name]
                if isinstance(param, np.ndarray):
                    param = _mx_np.array(param) if is_np_array() else nd.array(param)
                params[name]._load_init(param, ctx, cast_dtype=cast_dtype, dtype_source=dtype_source)

    def register_child(self, block, name=None):
        """Registers block as a child of self. :py:class:`Block` s assigned to self as
        attributes will be registered automatically."""
        if name is None:
            name = str(len(self._children))
        self._children[name] = weakref.ref(block)

    def register_forward_pre_hook(self, hook):
        r"""Registers a forward pre-hook on the block.

        The hook function is called immediately before :func:`forward`.
        It should not modify the input or output.

        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input) -> None`.

        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_pre_hooks, hook)
        return handle

    def register_forward_hook(self, hook):
        r"""Registers a forward hook on the block.

        The hook function is called immediately after :func:`forward`.
        It should not modify the input or output.

        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input, output) -> None`.

        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_hooks, hook)
        return handle

    def apply(self, fn):
        r"""Applies ``fn`` recursively to every child block as well as self.

        Parameters
        ----------
        fn : callable
            Function to be applied to each submodule, of form `fn(block)`.

        Returns
        -------
        this block
        """
        for cld in self._children.values():
            cld().apply(fn)
        fn(self)
        return self

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        """Initializes :py:class:`Parameter` s of this :py:class:`Block` and its children.

        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when :py:meth:`Parameter.init` is ``None``.
            Otherwise, :py:meth:`Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        verbose : bool, default False
            Whether to verbosely print out details on initialization.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
        params = self.collect_params()
        if verbose:
            init.set_verbosity(verbose=verbose)
        for v in params.values():
            v.initialize(None, ctx, init, force_reinit=force_reinit)

    def save(self, prefix):
        """Save the model architecture and parameters to load again later

        Saves the model architecture as a nested dictionary where each Block
        in the model is a dictionary and its children are sub-dictionaries.

        Each Block is uniquely identified by Block class name and a unique ID.
        We save each Block's parameter UUID to restore later in order to match
        the saved parameters.

        Recursively traverses a Block's children in order (since its an
        OrderedDict) and uses the unique ID to denote that specific Block.

        Assumes that the model is created in an identical order every time.
        If the model is not able to be recreated deterministically do not
        use this set of APIs to save/load your model.

        For HybridBlocks, the cached_graph is saved (Symbol & inputs) if
        it has already been hybridized.

        Parameters
        ----------
        prefix : str
            The prefix to use in filenames for saving this model:
            <prefix>-model.json and <prefix>-model.params
        """
        # create empty model structure
        model = {}
        def _save_cached_graphs(blk, structure, index=0):
            # create new entry for this block
            mdl = {}
            # encode unique name based on block type and ID
            name = type(blk).__name__.lower()
            structure[name+str(index)] = mdl
            index += 1
            if isinstance(blk, HybridBlock):
                if blk._cached_graph:
                    # save in/out formats
                    mdl['in_format'] = blk._in_format
                    mdl['out_format'] = blk._out_format
                    # save cached graph & input symbols
                    syms, out = blk._cached_graph
                    mdl_syms = []
                    for sym in syms:
                        mdl_syms.append(sym.tojson())
                    mdl['inputs'] = mdl_syms
                    mdl['symbol'] = out.tojson()
                    mdl['hybridized'] = True
                else:
                    mdl['hybridized'] = False
            # save param uuids
            pmap = {}
            mdl['params'] = pmap
            pnames = list(blk.params.keys())
            for p in pnames:
                param = blk.params[p]
                pmap[p] = param._uuid
            # recursively save children
            for child in blk._children.values():
                index = _save_cached_graphs(child(), mdl, index)
            # return latest index (ie. block count)
            return index

        # save top-level block
        _save_cached_graphs(self, model)
        # save model
        with open(prefix+'-model.json', 'w') as fp:
            json.dump(model, fp)
        # save params
        self.save_parameters('MyModel-model.params')

    def load(self, prefix):
        """Load a model saved using the `save` API

        Reconfigures a model using the saved configuration. This function
        does not regenerate the model architecture. It resets each Block's
        parameter UUIDs as they were when saved in order to match the names of the
        saved parameters.

        This function assumes the Blocks in the model were created in the same
        order they were when the model was saved. This is because each Block is
        uniquely identified by Block class name and a unique ID in order (since
        its an OrderedDict) and uses the unique ID to denote that specific Block.

        Assumes that the model is created in an identical order every time.
        If the model is not able to be recreated deterministically do not
        use this set of APIs to save/load your model.

        For HybridBlocks, the cached_graph (Symbol & inputs) and settings are
        restored if it had been hybridized before saving.

        Parameters
        ----------
        prefix : str
            The prefix to use in filenames for loading this model:
            <prefix>-model.json and <prefix>-model.params
        """
        # load model json from file
        with open(prefix+'-model.json') as fp:
            model = json.load(fp)

        def _load_cached_graphs(blk, structure, index=0):
            # get block name
            name = type(blk).__name__.lower()
            # lookup previous encoded name based on block type and ID
            mdl = structure[name+str(index)]
            index += 1
            if isinstance(blk, HybridBlock):
                if mdl['hybridized']:
                    # restore in/out formats
                    blk._in_format = mdl['in_format']
                    blk._out_format = mdl['out_format']
                    # get saved symbol
                    out = fromjson(mdl['symbol'])
                    syms = []
                    # recreate inputs for this symbol
                    for inp in mdl['inputs']:
                        syms.append(fromjson(inp))
                    # reset cached_graph and active status
                    blk._cached_graph = (syms, out)
                    blk._active = True
            # reload param uuids
            pmap = mdl['params']
            for p, uuid in pmap.items():
                param = blk.params[p]
                param._uuid = uuid
            # recursively reload children
            for child in blk._children.values():
                index = _load_cached_graphs(child(), mdl, index)
            # return latest index (ie. block count)
            return index

        # load top-level block
        _load_cached_graphs(self, model)
        # load params
        self.load_parameters('MyModel-model.params')

    def hybridize(self, active=True, **kwargs):
        """ Please refer description of HybridBlock hybridize().
        """
        for cld in self._children.values():
            cld().hybridize(active, **kwargs)

    def cast(self, dtype):
        """Cast this Block to use another data type.

        Parameters
        ----------
        dtype : str or numpy.dtype
            The new data type.
        """
        for child in self._children.values():
            child().cast(dtype)
        for _, param in self.params.items():
            param.cast(dtype)

    def zero_grad(self):
        """Sets all Parameters' gradient buffer to 0."""
        # collect gradient arrays for each ctx
        arrays = defaultdict(list)
        params = self.collect_params()
        for p in params.values():
            if p.grad_req == 'null' or p._grad is None:
                continue
            for g in p.list_grad():
                if g.stype == 'row_sparse':
                    ndarray.zeros_like(g, out=g)
                else:
                    if is_np_array():
                        arrays[g.ctx].append(g.as_nd_ndarray())
                    else:
                        arrays[g.ctx].append(g)

        if len(arrays) == 0:
            return

        for arr in arrays.values():
            ndarray.reset_arrays(*arr, num_arrays=len(arr))

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts.

        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        params = self.collect_params()
        for i in params.values():
            i.reset_ctx(ctx)

    def setattr(self, name, value):
        """Set an attribute to a new value for all Parameters.

        For example, set grad_req to null if you don't need gradient w.r.t a
        model's Parameters::

            model.setattr('grad_req', 'null')

        or change the learning rate multiplier::

            model.setattr('lr_mult', 0.5)

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : valid type for attribute name
            The new value for the attribute.
        """
        params = self.collect_params()
        for i in params.values():
            setattr(i, name, value)

    def share_parameters(self, shared):
        """Share parameters recursively inside the model.

        For example, if you want ``dense1`` to share ``dense0``'s weights, you can do::

            dense0 = nn.Dense(20)
            dense1 = nn.Dense(20)
            dense1.share_parameters(dense0.collect_params())

        which equals to
            dense1.weight = dense0.weight
            dense1.bias = dense0.bias

        Note that unlike the `load_parameters` or `load_dict` functions,
        `share_parameters` results in the `Parameter` object being shared (or
        tied) between the models, whereas `load_parameters` or `load_dict` only
        set the value of the data dictionary of a model. If you call
        `load_parameters` or `load_dict` after `share_parameters`, the loaded
        value will be reflected in all networks that use the shared (or tied)
        `Parameter` object.

        Parameters
        ----------
        shared : Dict
            Dict of the shared parameters.

        Returns
        -------
        this block
        """
        if shared is None:
            return self
        if not isinstance(shared, (dict, OrderedDict)):
            raise ValueError("'shared' should be in type of Dict. Get type {}!".format(type(shared)))
        shared_set = set(shared.keys())
        self._shared_parameters(shared, shared_set)
        if len(shared_set) > 0:
            for name in shared_set:
                warnings.warn("Parameter name {} is not in the current model!".format(name))
        return self

    def _shared_parameters(self, shared, shared_set, prefix=""):
        if prefix:
            prefix += '.'
        for name in self._reg_params:
            key = prefix + name
            if shared.get(key) is not None:
                setattr(self, name, shared[key])
                shared_set.remove(key)
        for name, child in self._children.items():
            child()._shared_parameters(shared, shared_set, prefix + name)

    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        for hook in self._forward_pre_hooks.values():
            hook(self, args)

        out = self.forward(*args)

        for hook in self._forward_hooks.values():
            hook(self, args, out)
        if _mx_npx.is_np_array():
            _check_all_np_ndarrays(out)
        return out

    def forward(self, *args):
        """Overrides to implement forward computation using :py:class:`NDArray`. Only
        accepts positional arguments.

        Parameters
        ----------
        *args : list of NDArray
            Input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError

    def register_op_hook(self, callback, monitor_all=False):
        """Install callback monitor.

        Parameters
        ----------
        callback : function
            Function called to inspect the values of the intermediate outputs
            of blocks after hybridization. It takes 3 parameters:
            name of the tensor being inspected (str)
            name of the operator producing or consuming that tensor (str)
            tensor being inspected (NDArray).
        monitor_all : bool, default False
            If True, monitor both input and output, otherwise monitor output only.
        """
        for cld in self._children.values():
            cld().register_op_hook(callback, monitor_all)

    def summary(self, *inputs):
        """Print the summary of the model's output and parameters.

        The network must have been initialized, and must not have been hybridized.

        Parameters
        ----------
        inputs : object
            Any input that the model supports. For any tensor in the input, only
            :class:`mxnet.ndarray.NDArray` is supported.
        """
        summary = OrderedDict()
        seen = set()
        hooks = []

        def _get_shape_str(args):
            def flatten(args):
                if not isinstance(args, (list, tuple)):
                    return [args], int(0)
                flat = []
                fmts = []
                for i in args:
                    arg, fmt = flatten(i)
                    flat.extend(arg)
                    fmts.append(fmt)
                return flat, fmts

            def regroup(args, fmt):
                if isinstance(fmt, int):
                    if fmt == 0:
                        return args[0], args[1:]
                    return args[:fmt], args[fmt:]
                ret = []
                for i in fmt:
                    res, args = regroup(args, i)
                    ret.append(res)
                return ret, args

            flat_args, fmts = flatten(args)
            flat_arg_shapes = [x.shape if isinstance(x, ndarray.NDArray) else x
                               for x in flat_args]
            shapes = regroup(flat_arg_shapes, fmts)[0]
            if isinstance(shapes, list):
                shape_str = str(shapes)[1:-1]
            else:
                shape_str = str(shapes)
            return shape_str.replace('L', '')

        def _register_summary_hook(block):
            assert not isinstance(block, HybridBlock) or not block._active, \
                    '"{}" must not be hybridized to print summary.'.format(type(block).__name__)
            def _summary_hook(block, _, outputs):
                class_name = block.__class__.__name__
                block_idx = len(summary) - 1

                m_key = '%s-%i' % (class_name, block_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['output_shape'] = _get_shape_str(outputs)

                params = 0
                summary[m_key]['trainable'] = 0
                summary[m_key]['shared'] = 0
                for p in block.params.values():
                    params += p.data().size
                    summary[m_key]['trainable'] += 0 if p.grad_req == 'null' else p.data().size
                    if p in seen:
                        summary[m_key]['shared'] += p.data().size
                    else:
                        seen.add(p)
                summary[m_key]['n_params'] = params

            from .nn.basic_layers import Sequential, HybridSequential
            if not isinstance(block, (Sequential, HybridSequential)):
                hooks.append(block.register_forward_hook(_summary_hook))

        summary['Input'] = OrderedDict()
        summary['Input']['output_shape'] = _get_shape_str(inputs)
        summary['Input']['n_params'] = 0
        summary['Input']['trainable'] = 0
        summary['Input']['shared'] = 0

        try:
            self.apply(_register_summary_hook)
            self(*inputs)

            line_format = '{:>20}  {:>42} {:>15}'
            print('-'*80)
            print(line_format.format('Layer (type)', 'Output Shape', 'Param #'))
            print('='*80)
            total_params = 0
            trainable_params = 0
            shared_params = 0
            for layer in summary:
                print(line_format.format(layer,
                                         str(summary[layer]['output_shape']),
                                         summary[layer]['n_params']))
                total_params += summary[layer]['n_params']
                trainable_params += summary[layer]['trainable']
                shared_params += summary[layer]['shared']
            print('='*80)
            print('Parameters in forward computation graph, duplicate included')
            print('   Total params: ' + str(total_params))
            print('   Trainable params: ' + str(trainable_params))
            print('   Non-trainable params: ' + str(total_params - trainable_params))
            print('Shared params in forward computation graph: ' + str(shared_params))
            print('Unique parameters in model: ' + str(total_params - shared_params))
            print('-'*80)
        finally:
            for h in hooks:
                h.detach()

