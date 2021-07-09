```{.python .input}
class Net(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.hidden1 = nn.Dense(256, activation='relu')
        self.a = gluon.Parameter('a', shape=(1, 2, 3, 4))
    
    def forward(self, x):
        ctx = x.ctx
        x1 = np.expand_dims(x, 0)
        x2 = x.reshape(-1, -2, -4)
        x3 = np.concatenate([self.a.data(ctx), x1], axis=1)
        x4 = np.dot(x2, x3.T)
        x5 = self.hidden1(x4)
        axes = list(range(x5.ndim))
        del axes[1]
        return mx.np.mean(x5, axis=axes)


input = mx.np.random.uniform(size=(2,3,4))
net = Net()
net.initialize()
net(input)
```