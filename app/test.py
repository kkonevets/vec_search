import numpy as np
import base64
import json
import requests
import argparse
import aiohttp
import asyncio
from main import RequestImages, RequestEmbeddings, do_post
import cv2
import glob
import tools


def image2base64(image_path, image_quality=90):
    np_img = cv2.imread(image_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    _, nparr = cv2.imencode('.jpg', np_img, encode_param)
    return base64.b64encode(nparr.tobytes()).decode()


async def test_images():
    headers = {"content-type": "application/json"}
    url = 'http://%s:%s/images' % (args.host, args.port)

    async with aiohttp.ClientSession() as session:
        request = RequestImages(distance_threshold=0.75, limit=5)
        it = glob.glob("../../data/avatars_examples/*.jpg")
        async for i, fname in tools.aiter(it):
            if i == 0:
                b64 = b'some invalid base64 data'
            else:
                b64 = image2base64(fname)
            request.data.append(b64)
        response = await do_post(session, request.json(), url, headers)
        print(response)


def load_test_vecs():
    npembs = np.load('../../data/xq_sample.npy')
    embs = []
    for v in npembs:
        embs.append(base64.b64encode(v))
    data = RequestEmbeddings(data=embs, distance_threshold=0.75, limit=5)
    return data.json()


async def test_embeddings():
    data = load_test_vecs()
    url = 'http://%s:%s/embeddings' % (args.host, args.port)
    headers = {"content-type": "application/json"}

    async with aiohttp.ClientSession() as session:
        post_tasks = []
        async for _, i in tools.aiter(range(20)):
            post_tasks.append(do_post(session, data, url, headers))
        for text in await asyncio.gather(*post_tasks):
            try:
                print(i, json.loads(text))
            except:
                print(text)


# python test.py images --host server2.company --port 8123
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('method',
                        choices=['embeddings', 'images'],
                        default='images')
    parser.add_argument('--host', default='server2.company')
    parser.add_argument('--port', default='8123')

    args = parser.parse_args()

    if args.method == 'embeddings':
        coro = test_embeddings()
    elif args.method == 'images':
        coro = test_images()

    asyncio.run(coro)