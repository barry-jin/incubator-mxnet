import mxnet
import numpy as _np
from mxnet import np, npx
from mxnet.ndarray import NDArray
from mxnet import _api_internal
npx.set_np()

def test2():
    x = np.array([1, 2, 3])
    y = x.asnumpy()
    print(y)

def test_empty():
    amap = mxnet._ffi.convert_to_node({})
    adt = mxnet._ffi.convert_to_node(())
    assert len(amap) == 0
    assert len(adt) == 0

def test_str_map():
    amap = mxnet._ffi.convert_to_node({"a": 2, "b": 3})
    # amap = mxnet._ffi.convert_to_node({"a": "c", "b": "d"})
    assert "a" in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    # TODO
    # assert amap["a"].value == 2
    assert "a" in dd
    assert "b" in dd

# TODO
def test_adt():
    a = mxnet._ffi.convert_to_node([1, 2, 3])
    assert len(a) == 3
    assert a[-1].value == 3
    a_slice = a[-3:-1]
    assert (a_slice[0].value, a_slice[1].value) == (1, 2)

def test_ndarray_container():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    arr = mxnet._ffi.convert_to_node([x, y])
    assert _np.array_equal(arr[0].asnumpy(), x.asnumpy())
    assert isinstance(arr[0], NDArray)
    amap = mxnet._ffi.convert_to_node({'x': x, 'y': y})
    assert "x" in amap
    assert _np.array_equal(amap["y"].asnumpy(), y.asnumpy())
    assert isinstance(amap["y"], NDArray)
    dd = dict(amap.items())
    print(dd)

def test_string():
    s = mxnet.container.String("xyz")
    assert isinstance(s, mxnet.container.String)
    assert isinstance(s, str)
    assert s.startswith("xy")
    assert s + "1" == "xyz1"
    y = _api_internal._echo(s)
    assert isinstance(y, mxnet.container.String)
    assert s.__mxnet_object__.same_as(y.__mxnet_object__)
    assert s == y

def test_string_adt():
    s = mxnet.container.String("xyz")
    arr = mxnet._ffi.convert_to_node([s, s])
    assert arr[0] == s
    assert isinstance(arr[0], mxnet.container.String)

def add_two(a, b, c):
    return a+b+c

def test_ffi_map():
    amap = mxnet._ffi.convert_to_node({"a": 2, "b": 3})

    print(mxnet._ctypes._api_internal.MapVSum(amap))

def test_split():
    z = np.ones((4,4))
    print(np.split(z, 4))

def test_kvstore_init(stype='default'):
    shape = (4, 4)
    keys = [5, 7, 11]
    str_keys = ['b', 'c', 'd']
    kv = mxnet.kv.create()
    a = mxnet.nd.ones(shape)
    b = mxnet.nd.zeros(shape)
    kv.init('1', mxnet.nd.zeros(shape))
    kv.push('1', [a,a,a,a])
    kv.pull('1', b)
    print(b)

if __name__ == '__main__':
    # test2()
    # test_empty()
    # test_str_map()
    # test_adt()
    # test_ndarray_container()
    # test_string()
    # test_string_adt()
    test_ffi_map()
    # test_split()
    # test_kvstore_init()
