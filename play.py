import faiss
import zipfile
import array
import app.tools as tools
import time

import numpy as np
import json
import requests


def image_batch_example():
    image_batch = np.random.rand(1, 160, 160, 3).astype('f')

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://server2.company:8501/v1/models/facenet:predict',
        data=json.dumps(
            {
                "inputs": {
                    "image_batch": image_batch.tolist(),
                    "phase_train": False
                },
            },
            separators=(',', ':')),
        headers=headers)
    resp = json.loads(json_response.text)
    # print(resp)
    return resp['outputs']


def zip2bin(fname, fout):
    try:
        with zipfile.ZipFile(fname) as z:
            with z.open(z.namelist()[0]) as f:
                next(f)  # skip header
                with open(fout, 'wb') as out:
                    for line in f:
                        it = map(float, line.split(b';')[1:])
                        array.array('f', it).tofile(out)

    except zipfile.BadZipFile:
        print('Error: Zip file is corrupted')


def flat():
    index = faiss.IndexFlatL2(ncols)

    for i, xb in enumerate(it):
        if i == 0:
            xq = xb[:test_size]
        index.add(xb)

    D, I = index.search(xq, 2)

    with open('../data/faiss/test_index.bin', 'wb') as f:
        I = array.array('L', I[:, 1])
        I.tofile(f)


def voronoi_gpu():
    test_index = tools.load_vector('../data/adamskij/test_index.bin', 'L')

    nlist = 100
    quantizer = faiss.IndexFlatL2(ncols)
    cpu_index = faiss.IndexIVFFlat(quantizer, ncols, nlist)

    xb = tools.load_2d_vec(fout, ncols, typecode='f')
    xq = np.copy(xb[:test_size])
    cpu_index.train(xb)

    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)

    ress = []
    for i in range(ngpus):
        res = faiss.StandardGpuResources()
        if i in (2, 3, 4, 5):
            res.noTempMemory()
        res.initializeForDevice(i)
        ress.append(res)

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(ress, cpu_index, co)
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)

    gpu_index.add(xb[:20_000_000])
    # for xb in it:
    #     gpu_index.add(xb)

    for i in range(20):
        gpu_index.nprobe = i + 1  # default nprobe is 1, try a few more
        start_time = time.time()

        D, I = gpu_index.search(xq, 2)

        secs = time.time() - start_time
        # acc = (I[:, 1] == test_index).sum()
        print(i + 1, secs)


def voronoi():
    test_index = tools.load_vector('../data/adamskij/test_index.bin', 'L')

    nlist = 100
    quantizer = faiss.IndexFlatL2(ncols)
    index = faiss.IndexIVFFlat(quantizer, ncols, nlist)

    xb = tools.load_2d_vec(fout, ncols, typecode='f')
    xq = np.copy(xb[:test_size])
    index.train(xb)

    for xb in it:
        index.add(xb)

    index.nprobe = 12
    lims, D, I = index.range_search(xq, 0.75)
    for i in range(len(xq)):
        dists = D[lims[i]:lims[i + 1]]
        sorted_ixs = dists.argsort()
        dists = dists[sorted_ixs]
        sub_I = I[lims[i]:lims[i + 1]][sorted_ixs]

    for i in range(20):
        index.nprobe = i + 1  # default nprobe is 1, try a few more
        start_time = time.time()

        D, I = index.search(xq, 2)

        secs = time.time() - start_time
        acc = (I[:, 1] == test_index).sum()
        print(i + 1, acc, secs)


if __name__ == "__main__":
    fout = '../data/faiss/vectors.bin'
    # zip2bin('../data/faiss/embs.zip', fout)

    ncols = 512
    test_size = 100

    it = tools.fromfile_iter(fout, ncols, chunk_size=10**8, typecode='f')
