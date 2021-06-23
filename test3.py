from mxnet import autograd as ag
from mxnet import np
x = np.array([1,2,3,4])
x.attach_grad()
y = np.array([5,6,7,8])
y.attach_grad()

ag.set_recording(True)
u = x * y
v = u.detach()
v.attach_grad()
z = v * x
ag.set_recording(False)
z.backward()
u.backward(v.grad)
print(x.grad, y.grad)
