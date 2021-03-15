import mxnet as mx
import ctypes
from mxnet import np, npx, _global_var
npx.set_np()

# def batchify_fn(data):
#     out = np.empty((len(data),) + data[0].shape, dtype=data[0].dtype)
#     return np.stack(data, out=out)

def batchify_fn(data):
    # return _global_var._np_ndarray_cls(ctypes.c_void_p(data))
    # print(type(data))
    return data

def test_ffi_batchify():
    # data = []
    # for i in range(10):
    #     data.append(np.zeros((2,2)))
    print(mx._ctypes._api_internal.TestBatchify(batchify_fn, np.zeros((2,2))))

test_ffi_batchify()

