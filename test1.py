from mxnet import autograd as ag
from mxnet import np
# x = np.array([1, 2, 3, 4])
# x.attach_grad()
# y = np.array([5, 6, 7, 8])
# y.attach_grad()

# ag.set_recording(True)
# u = x * y
# # u.attach_grad()
# z = 5 * u
# ag.set_recording(False)
# z.backward()
# print(x.grad, y.grad, u.grad)
# x = np.ones(shape=(2,2))*2
# x.attach_grad()
# with ag.record():
#     y = 2 * np.exp(x) + 3 * np.square(x)
# y.backward()
# print(x.grad)

from jax import jvp

from jax import vjp
import jax.numpy as jnp

def vgrad(f, x):
  y, vjp_fn = vjp(f, x, x)
  return vjp_fn(jnp.ones(y.shape))[0]

print(vgrad(lambda x: 3*x**2, jnp.ones((2, 2))))


