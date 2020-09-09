import logging
import socketserver
import multiprocessing
import gc
import numpy as np
import tools
from messaging import send_msg, recv_msg, MessageType
import time
import tcpclient
import bson
import faiss
from main import Response, SearchResult
from tools import HOST, PORT
import os
import traceback
import config

VEC_PATH = '../../data/faiss/vectors.bin'
IDS_PATH = '../../data/faiss/face_ids.csv'

# os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count()/2))
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

ferrlog = config.logconfig_dict['handlers']['error_console']['filename']
logging.basicConfig(filename=ferrlog, level=logging.ERROR)


def load_index():
    ncols = 512
    nlist = 100

    quantizer = faiss.IndexFlatL2(ncols)
    index = faiss.IndexIVFFlat(quantizer, ncols, nlist)

    xb = tools.load_2d_vec(VEC_PATH, ncols, typecode='f')
    index.train(xb)

    it = tools.fromfile_iter(VEC_PATH,
                             ncols,
                             chunk_size=10**8,
                             typecode='f',
                             desc='building index')
    for xb in it:
        index.add(xb)

    index.nprobe = 12

    # face_ids = []
    # with open(IDS_PATH) as f:
    #     for line in f:
    #         face_ids.append(line)

    # if len(face_ids) != nvecs:
    #     logging.error(
    #         "number of face ids and number of vectors differ %d != %d" %
    #         (len(face_ids), nvecs))
    #     raise AssertionError

    logging.info("index loaded, %d items" % (index.ntotal))

    return index, None


def worker():
    batch = bytearray()
    while True:
        time.sleep(1)
        batch += np.random.rand(512).astype('f').tobytes()
        if len(batch) == 5 * 512 * 4:
            data = {'data': bytes(batch), 'nrows': 5}
            tcpclient.send(HOST, PORT, data, MessageType.BATCH, False)
            batch = bytearray()


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        msg = recv_msg(self.request)
        data = bson.BSON.decode(msg)

        if data['type'] == MessageType.FRONTEND.value:
            # time.sleep(5)
            response = Response()
            try:
                xq = tools.load_2d_buf(data['data'], data['nrows'], 'f')
                assert xq is not None

                lims, D, I = index.range_search(xq, data['distance_threshold'])
                for i in range(len(xq)):
                    dists = D[lims[i]:lims[i + 1]]
                    sorted_ixs = dists.argsort()[:data['limit']]
                    dists = dists[sorted_ixs]
                    sub_I = I[lims[i]:lims[i + 1]][sorted_ixs]

                    res = SearchResult(face_ids=["blabla@vk.com"] * len(dists),
                                       distances=dists.tolist())
                    response.result.append(res)
            except AssertionError:
                response.detail = 'Incorrect data shape'
                logging.error(traceback.format_exc())
            except:
                response.detail = 'Internal Server Error'
                logging.error(traceback.format_exc())

            send_msg(self.request, bson.BSON.encode(response.dict()))
        else:
            batch = tools.load_2d_buf(data['data'], data['nrows'], 'f')
            # print(batch.shape)


if __name__ == "__main__":
    index, face_ids = load_index()

    reader = multiprocessing.Process(target=worker)
    reader.daemon = True
    reader.start()

    socketserver.TCPServer.allow_reuse_address = True
    socketserver.TCPServer.request_queue_size = 10
    with socketserver.TCPServer((HOST, PORT), TCPHandler) as server:
        print('server started')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
            reader.terminate()
