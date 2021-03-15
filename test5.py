import mxnet as mx

def test_op_hook_output_names():
    def check_name(block, expected_names, inputs=None, expected_opr_names=None, monitor_all=False):
        opr_names = []
        output_names = []

        def mon_callback(node_name, opr_name, arr):
            output_names.append(node_name)
            opr_names.append(opr_name)
            assert isinstance(arr, mx.nd.NDArray)

        block.register_op_hook(mon_callback, monitor_all)
        if not inputs:
            block(mx.nd.ones((2, 3, 4)))
        else:
            block(inputs)

        for output_name, expected_name in zip(output_names, expected_names):
            print(output_name)
            assert output_name == expected_name

        if expected_opr_names:
            for opr_name, expected_opr_name in zip(opr_names, expected_opr_names):
                assert opr_name == expected_opr_name

    # Test with Dense layer
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Dense(2))
    model.initialize()
    model.hybridize()
    check_name(model, ["hybridsequential_dense0_fwd_output"])

    # Test with Activation, FListInputNames not registered, input name will have _input appended
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Activation("relu"))
    model.initialize()
    model.hybridize()
    check_name(model, ["hybridsequential_activation0_fwd_output"])

    # Test with Pooling, monitor_all is set to True
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.AvgPool1D())
    model.initialize()
    model.hybridize()
    check_name(model, ['hybridsequential_avgpool1d0_fwd_data', 'hybridsequential_avgpool1d0_fwd_output'],
               expected_opr_names=["Pooling"], monitor_all=True)

    # stack two layers and test
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Dense(2))
    model.add(mx.gluon.nn.Activation("relu"))
    model.initialize()
    model.hybridize()
    check_name(model,
               ['hybridsequential_dense0_fwd_data', 'hybridsequential_dense0_fwd_weight',
                'hybridsequential_dense0_fwd_bias', 'hybridsequential_dense0_fwd_output',
                'hybridsequential_activation0_fwd_input0', 'hybridsequential_activation0_fwd_output'], monitor_all=True)

    # check with different hybridize modes
    model.hybridize(static_alloc=True)
    check_name(model,
               ['hybridsequential_dense0_fwd_data', 'hybridsequential_dense0_fwd_weight',
                'hybridsequential_dense0_fwd_bias', 'hybridsequential_dense0_fwd_output',
                'hybridsequential_activation0_fwd_input0', 'hybridsequential_activation0_fwd_output'], monitor_all=True)

if __name__ == '__main__':
    test_op_hook_output_names()
