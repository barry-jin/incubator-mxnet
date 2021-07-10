```{.python .input}
adagrad_optimizer = optimizer.AdaGrad(learning_rate=0.1, eps=1e-07)
rmsprop_optimizer = optimizer.RMSProp(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False)
trainer = gluon.Trainer(net.collect_params(), optimizer=adagrad_optimizer)
```