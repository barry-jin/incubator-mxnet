import mxnet as mx
import struct
import timeit
import numpy as _np
from collections import namedtuple
from mxnet import np, npx 
npx.set_np()


setup = """
from mxnet import np, npx
npx.set_np()
seq_length = 10
batch_size = 4
input_size = 4
state_size = 1
data = np.random.normal(0, 1, (seq_length, batch_size, input_size))
state = np.random.normal(0, 1, (1, batch_size, state_size))
p = np.random.normal(0, 1, ((input_size + state_size + 2)*state_size),)
"""
stmt = """
out = npx.rnn(data=data, parameters=p, mode='rnn_tanh', \
              state=state, state_size=state_size, num_layers=1, \
              bidirectional=False, state_outputs=False, p=0.0, \
              use_sequence_length=False, projection_size=None, \
              lstm_state_clip_min=None,lstm_state_clip_max=None, \
              lstm_state_clip_nan=False)
"""
timer = timeit.Timer(setup=setup,
                     stmt=stmt)
num_repeat = 1000
# print(min(timer.repeat(repeat=10, number=num_repeat)) / num_repeat)



from mxnet import np, npx
npx.set_np()
inp = np.ones((2, 10))
gamma=np.ones((2))
beta=np.zeros((2))


out = npx.layer_norm(inp, gamma=gamma, beta=beta, axis=0, eps=1e5, output_mean_var=False)

print(out)
