import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet import autograd as ag, npx

# class Block1(HybridBlock):
#     def __init__(self):
#         super(Block1, self).__init__()

#     def forward(self, a, b):
#         return a - b

# class Block2(HybridBlock):
#     def __init__(self):
#         super(Block2, self).__init__()

#     def forward(self, a, b):
#         return npx.cond(lambda a, b: a > b,
#                         lambda a, b: a,
#                         lambda a, b: b,
#                         [a, b])

# class Block3(HybridBlock):
#     def __init__(self):
#         super(Block3, self).__init__()

#     def forward(self, a, b):
#         return a * b

# class MixedBlock(HybridBlock):
#     def __init__(self, block1, block2, block3):
#         super(MixedBlock, self).__init__()
#         self.block1 = block1
#         self.block2 = block2
#         self.block3 = block3

#     def forward(self, a, b):
#         out1 = self.block1(a, b)
#         out2 = self.block2(out1, b)
#         out3 = self.block3(out2, a)
#         return out3

# block1, block2, block3 = Block1(), Block2(), Block3()
# mix = MixedBlock(block1, block2, block3)
# mix.initialize()
# mix.hybridize()
# # block1.hybridize()
# # block3.hybridize()
# block2.hybridize(active=False)
# x_list = [mx.np.array([i]) for i in range(5)]
# y_list = [mx.np.array([1])]*5
# for x, y in zip(x_list, y_list):
#     x.attach_grad()
#     y.attach_grad()
#     with ag.record():
#         out = mix(x, y)
#     out.backward()
#     print(out, x.grad, y.grad)

# class AddBlock(HybridBlock):
#     def __init__(self):
#         super(AddBlock, self).__init__()
    
#     def forward(self, a, b):
#         return a + b

# add = AddBlock()
# add.initialize()
# add.hybridize(static_alloc=True)

# x = mx.np.array([0.4])
# y = mx.np.array([0.5])
# x.attach_grad(grad_req='write')
# y.attach_grad(grad_req='null')
# with ag.record():
#     out = add(x, y)
# out.backward()
# print("\nINPUT 1: {}\nINPUT 2: {}\nOUTPUT: {}\nGRAD 1: {}\n"
#     "GRAD 2: {}\n".format(x, y, out, x.grad, y.grad))

# cond = lambda i, s: i <= 5
# func = lambda i, s: ([i + s], [i + 1, s + i])
# loop_vars = (mx.np.array([0], dtype="int64"), mx.np.array([1], dtype="int64"))
# outputs, states = mx.npx.while_loop(cond, func, loop_vars, max_iterations=10)
# print(outputs)
# print(states)

# cond = lambda i, s: i <= 5
# func = lambda i, s: ([i + s], [i + 1, s + i])
# loop_vars = (mx.sym.var('i'), mx.sym.var('s'))
# outputs, states = mx.sym.contrib.while_loop(cond, func, loop_vars, max_iterations=10)

# class step(HybridBlock):
#     def forward(self, data, states):
#         return (data + states[0], [states[0] * 2])
# data = mx.np.random.uniform(size=(2, 10))
# states = [mx.np.random.uniform(size=(10))]
# data.attach_grad()
# states[0].attach_grad()
# with ag.record():
#     outs, states = npx.control_flow.foreach(step(), data, states)
# outs.backward()
# print(outs, states)


# step = lambda data, states: (data + states[0], [states[0] * 2])
# data = mx.sym.var('data')
# states = [mx.sym.var('state')]
# outs, states = mx.sym.contrib.foreach(step, data, states)

# class SimpleBlock(HybridBlock):
#     def __init__(self):
#         self.conv1 = nn.Conv2D(64, 3)
#         self.conv2 = nn.Conv2D(128, 3)

#     def forward(self, x):
#         N, C, H, W = x.shape
#         if C == 64:
#             out = self.conv1(x)
#         else:
#             out = self.conv2(x)
#         return out

# class ConcatScaleAlign(nn.HybridBlock):
#     def __init__(self, **kwargs):
#         super(ConcatScaleAlign, self).__init__(**kwargs)
#         self.shared_weight = mx.gluon.Parameter('shared_weight', shape=(64, 4, 3, 3),
#                                                 init=mx.init.Xavier(magnitude=2.24),
#                                                 dtype='float32', allow_deferred_init=True)

#     def forward(self, x):
#         conv1 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
#                                    weight=self.shared_weight.data(x.ctx), no_bias=True)
#         conv2 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
#                                    weight=self.shared_weight.data(x.ctx)*2, no_bias=True)
#         return mx.np.concatenate([conv1, conv2], axis=1)

# concat = ConcatScaleAlign()
# concat.initialize(init=mx.init.Normal(0.5), force_reinit=True)
# data = mx.np.random.uniform(-1, 1.0, size=(64, 4, 10, 10), dtype='float32', ctx=mx.current_context())

# outputs = concat(data)

# calib_data = mx.gluon.data.DataLoader(data, batch_size=1)
# qnet = quantization.quantize_net(concat,
#                                 ctx=mx.current_context(),
#                                 exclude_layers=None,
#                                 exclude_operators=None,
#                                 quantized_dtype='int8',
#                                 calib_mode='naive',
#                                 calib_data=calib_data,
#                                 num_calib_batches=1,
#                                 quantize_mode='full',
#                                 quantize_granularity='tensor-wise')
# qsym, _ = qnet.export(None)
# init = False
# for k, v in qsym.attr_dict().items():
#     if k.find('quantized_sg_mkldnn_conv') != -1:
#         assert 'min_calib_range' in v
#         assert 'max_calib_range' in v
#         if not init:
#             min_calib_range = v['min_calib_range']
#             max_calib_range = v['max_calib_range']
#             init = True
#         else:
#             assert min_calib_range == v['min_calib_range']
#             assert max_calib_range == v['max_calib_range']

# class FCEltwise(nn.HybridBlock):
#     def __init__(self, use_bias, flatten, **kwargs):
#         super(FCEltwise, self).__init__(**kwargs)
#         self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten,
#                          weight_initializer=None)

#     def forward(self, x):
#         fc_out = self.fc(x)
#         out = mx.np.square(fc_out)
#         return out

# attrs = {'fc': {'with_eltwise': 'true'}}
# net = FCEltwise(True, True)

# net.initialize()
# net.hybridize(static_alloc=False, static_shape=False)
# data = mx.np.random.uniform(size=(64, 4, 10, 10), dtype='float32', ctx=mx.current_context())
# net(data)
# sym, params = net.export(None)


# sym_sg = sym.optimize_for('MKLDNN', dedup_subgraph=True, skip_infer=True)
# for name, attrs in attrs.items():
#     if len(attrs):
#         found = False
#         for k, v in sym_sg.attr_dict().items():
#             if k.find('sg_mkldnn_fully_connected') != -1:
#                 found = True
#                 for attr_name, attr_value in attrs.items():
#                     assert v[attr_name].lower() == attr_value.lower()
#         assert found


# class WhileLayer1(gluon.HybridBlock):
#     def __init__(self):
#         super(WhileLayer1, self).__init__()

#     def forward(self, inputs, states):
#         def cond(state1, state2):
#             s = mx.np.squeeze(mx.npx.slice(state1, begin=0, end=1))
#             return s == s
#         def step(state1, state2):
#             return state1 + 1, [state1 + 1, state2 + 1]
#         states = [states[0], states[0] + 1]
#         out1, states1 = mx.npx.while_loop(cond, step, states, max_iterations=5)
#         # The input variables have the same symbol name.
#         out, states = mx.npx.while_loop(cond, step, states1, max_iterations=5)
#         return out

# data = mx.np.random.normal(loc=0, scale=1, size=(2, 5))
# states = mx.np.random.normal(loc=0, scale=1, size=(5))

# data.attach_grad()
# states.attach_grad()
# layer = WhileLayer1()
# layer.initialize(ctx=mx.context.current_context())
# layer.hybridize()
# res1 = layer(data, [states])

# with mx.autograd.record():
#     res1 = layer(data, [states])
# res1.backward()
# print(res1)

# print(states)
# layer.initialize(ctx=mx.context.current_context())
# layer.hybridize()
# res2 = layer(data, [states])
# with mx.autograd.record():
#     res2 = layer(data, [states])
# print(res2)


# a, b = mx.np.array([1]), mx.np.array([2])
# pred = lambda a, b: a * b < 5
# then_func = lambda a, b: (a + 5) * (b + 5)
# else_func = lambda a, b: (a - 5) * (b - 5)
# outputs = mx.npx.cond(pred, then_func, else_func, [a, b])
# print(outputs[0])


# class RNNLayer(gluon.HybridBlock):
#     def __init__(self, cell_type, hidden_size, layout):
#         super(RNNLayer, self).__init__()
#         self.cell = cell_type(hidden_size)
#         self.layout = layout

#     def forward(self, inputs, states, valid_length):
#         if isinstance(valid_length, list) and len(valid_length) == 0:
#             valid_length = None
#         return gluon.rnn.rnn_cell.dynamic_unroll(self.cell, inputs, states,
#                                                     valid_length=valid_length,
#                                                     layout=self.layout)
    
#     def infer_shape(self, x, *args):
#         self.cell.infer_shape(0, x, False)

# batch_size = 2
# input_size = 2
# hidden_size = 20
# seq_len = 10
# ctx = mx.context.current_context()
# rnn_data = mx.np.random.normal(loc=0, scale=1, size=(seq_len, batch_size, input_size), ctx=ctx)
# valid_length = mx.np.round(mx.np.random.uniform(low=1, high=10, size=(batch_size), ctx=ctx))
# state_shape = (batch_size, hidden_size)
# states = [mx.np.random.normal(loc=0, scale=1, size=state_shape, ctx=ctx) for i in range(1)]

# cell = gluon.rnn.RNNCell(hidden_size)
# cell.infer_shape(0, rnn_data[0], False)
# cell.initialize(ctx=ctx)
# cell(rnn_data[0], states)
# params1 = cell.collect_params()
# orig_params1 = copy.deepcopy(params1)

# trainer = gluon.Trainer(params1, 'sgd', {'learning_rate' : 0.03})
# with mx.autograd.record():
#     res1, states1 = cell.unroll(seq_len, rnn_data, states, valid_length=valid_length,
#                                 layout='TNC', merge_outputs=True)
# res1.backward()
# trainer.step(batch_size)

# configs = [
#         lambda layer: None,
#         lambda layer: layer.hybridize(),
#         lambda layer: layer.hybridize({'inline_limit': 0}),
#         lambda layer: layer.hybridize({'static_alloc': True}),
#         lambda layer: layer.hybridize({'static_alloc': True, 'static_shape': True}) ]
# # configs = [lambda layer: None]
# # We can't pass None to a hybrid block, but it accepts an empty list.
# # so we use an empty list to represent valid_length if it's None.
# if valid_length is None:
#     valid_length = []
# for config in configs:
#     layer = RNNLayer(gluon.rnn.RNNCell, hidden_size, 'TNC')
#     layer.infer_shape(rnn_data)
#     layer.initialize(ctx=ctx)
#     config(layer)
#     res2, states2 = layer(rnn_data, states, valid_length)
#     params2 = layer.collect_params()
#     for key, val in orig_params1.items():
#         params2['cell.' + key].set_data(copy.deepcopy(val.data()))

#     trainer = gluon.Trainer(params2, 'sgd', {'learning_rate' : 0.03})
#     with mx.autograd.record():
#         res2, states2 = layer(rnn_data, states, valid_length)
#     assert_almost_equal(res1, res2, rtol=0.001, atol=0.0001)
#     assert len(states1) == len(states2)
#     res2.backward()
#     trainer.step(batch_size)

#     for key, val in params1.items():
#         weight1 = val.data()
#         weight2 = params2['cell.' + key].data()
#         print(weight1)
#         print(weight2)
#         assert_almost_equal(weight1, weight2, rtol=0.1, atol=0.01)

