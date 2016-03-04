from __future__ import division, print_function
from nose.tools import with_setup, assert_raises, assert_equal

import numpy as np


from lavaburst import utils


def test_compress_restore_path():
    # simple example, no special cases
    data = np.fromiter('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', dtype='S1')
    passmask = np.ones(52, dtype=bool)
    passmask[10:15] = False
    passmask[30:35] = False
    uncompressed_path = np.array([0, 25, 40, 50, 52]) #includes end node
    compressed_path   = np.array([0, 20, 30, 40, 42])

    assert np.all(utils.mask_compress_path(uncompressed_path, passmask) == compressed_path)
    assert np.all(utils.mask_restore_path(compressed_path, passmask) == uncompressed_path)


def test_masked_normalize():
    data = np.ones((10,10))
    passmask = [1,1,1,0,0,0,1,1,1,0]

    for ax in (0, 1):
        data_norm = utils.masked_normalize(data, passmask, axis=ax)
        sums = data_norm.sum(axis=ax)

        for i in (0,1,2,6,7,8):
            assert np.isclose(sums[i], 1.0)
        for i in (3,4,5,9):
            assert np.isclose(sums[i], 10.0)


def test_masked_ufunc():
    data = np.ones((10,10))
    passmask = [1,1,1,0,0,0,1,1,1,0]

    masked_add = utils.masked_ufunc(np.add, passmask)
    res = masked_add(data, 1, out=data.copy())

    for i in (0,1,2,6,7,8):
        assert np.allclose(res[i,:], 2.0)
        assert np.allclose(res[:,i], 2.0)

    assert np.allclose(res[np.ix_((3,4,5,9),(3,4,5,9))], 1.0)
