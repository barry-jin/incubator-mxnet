from mxnet import np, npx, gluon
npx.set_np()

class _Dataset(gluon.data.Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, key):
        return np.full((10,), key)

def test_multi_worker_shape():
    for thread_pool in [True]:
        batch_size = 1024
        shape = (batch_size+1, 11, 12)

        data = gluon.data.ArrayDataset(np.ones(shape))
        loader = gluon.data.DataLoader(
            data, batch_size=batch_size, num_workers=5, last_batch='keep', thread_pool=thread_pool)
        for batch in loader:
            if shape[0] > batch_size:
                assert batch.shape == (batch_size, shape[1], shape[2])
                shape = (shape[0] - batch_size, shape[1], shape[2])
            else:
                assert batch.shape == shape


test_multi_worker_shape()
