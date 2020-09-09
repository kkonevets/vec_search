import numpy as np
import os
import array
from tqdm import tqdm
from typing import Iterable
import base64

HOST, PORT = 'localhost', 5030


def fromfile_iter(fname, ncols, chunk_size=10**8, typecode='I', desc=None):
    fsize = os.path.getsize(fname)
    itemsize = array.array(typecode).itemsize
    row_size = (itemsize * ncols)
    assert fsize % row_size == 0
    total = fsize // itemsize
    chunk_size = chunk_size - chunk_size % row_size
    with open(fname, 'rb') as f:
        for i in tqdm(range(0, total, chunk_size), desc=desc):
            dif = total - i
            n = chunk_size if dif >= chunk_size else dif
            buff = array.array(typecode)
            buff.fromfile(f, n)
            a = to_numpy(buff, typecode, (len(buff) // ncols, ncols))
            yield a


def fromfile(a, fname, chunk_size=10**8):
    fsize = os.path.getsize(fname)
    assert fsize % a.itemsize == 0
    total = fsize // a.itemsize
    with open(fname, 'rb') as f:
        for i in range(0, total, chunk_size):
            dif = total - i
            n = chunk_size if dif >= chunk_size else dif
            a.fromfile(f, n)


def load_vector(fname, typecode='I'):
    a = array.array(typecode)
    fromfile(a, fname)
    return a


def to_numpy(buff, typecode=None, shape=None, order='C'):
    if shape is None:
        shape = len(buff)

    a = np.ndarray(shape,
                   buffer=memoryview(buff),
                   order=order,
                   dtype=buff.typecode)

    if typecode is None:
        typecode = buff.typecode
    if typecode != buff.typecode:
        a = a.astype(typecode)

    return a


def load_2d_vec(fname, ncols, typecode='L', order='C'):
    """
    buff=array.array('i', [1,2,3,4,5,6])

    order='C'
    v1 = [1,2,3]
    v2 = [4,5,6]

    order='F'
    v1 = [1,3,5]
    v2 = [2,4,6]

    a = [v1, v2]
    """

    buff = load_vector(fname, typecode=typecode)
    assert len(buff) % ncols == 0
    a = to_numpy(buff, typecode, (len(buff) // ncols, ncols), order)
    return a


def load_2d_buf(buff, nrows, typecode='f', order='C'):
    itemsize = array.array(typecode).itemsize
    rowsize = nrows * itemsize
    if len(buff) == 0 or len(buff) % rowsize != 0:
        return None
    shape = (nrows, len(buff) // rowsize)
    a = np.ndarray(shape, buffer=memoryview(buff), order=order, dtype=typecode)
    return a


def bytes_concat(bytes_list: Iterable[bytes]):
    ret = bytearray()
    for bs in bytes_list:
        ret += bs
    return bytes(ret)


async def aiter(iterable):
    for i, v in enumerate(iterable):
        yield i, v
