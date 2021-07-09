```{.python .input}
class Net(nn.HybridBlock):
    def __init__(self, prefix=None, **kwargs):
        super(Net, self).__init__(prefix=prefix, **kwargs)
        with self.name_scope():
            self.hidden1 = nn.Dense(256, activation='relu')
            self.a = self.params.get('a', shape=(1, 2, 3, 4))
    
    def hybrid_forward(self, F, x, a):
        x1 = x.expand_dims(0)
        x2 = x.reshape(-1, 0, -2)
        x3 = F.concat(a, x1)
        x4 = F.dot(x2, x3.T)
        x5 = self.hidden1(x4)
        return F.mean(x5, axis=1, exclude=1)


input = mx.random.uniform(shape=(2,3,4))
net = Net(prefix="net")
net.initialize()
net(input)
```