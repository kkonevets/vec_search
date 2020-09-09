import logging
from fastapi import FastAPI, HTTPException
from typing import Optional, List
import traceback
from pydantic import BaseModel, ValidationError
import base64
import tcpclient
import tools
from tools import HOST, PORT
from messaging import MessageType
import asyncio
import aiohttp
import time

app = FastAPI()

IMAGE_URL = 'http://tchaikovsky.company:8001/v1/faces'


class BaseRequest(BaseModel):
    data: List[str] = []
    distance_threshold: Optional[float] = 0.75
    limit: Optional[int] = 5


class RequestImages(BaseRequest):
    pass


class RequestEmbeddings(BaseRequest):
    pass


class SearchResult(BaseModel):
    detail: Optional[str] = ""
    face_ids: List[str] = []
    distances: List[float] = []


class EmbeddingsResponse(BaseModel):
    detail: Optional[str] = ""
    result: List[SearchResult] = []


class ImagesResponse(BaseModel):
    detail: Optional[str] = ""
    result: List[List[SearchResult]] = []


class RequestDetect(BaseModel):
    type: Optional[str] = 'imagebase64'
    value: str


class FaceEmbedding(BaseModel):
    version: Optional[str]
    data: str


class FaceElement(BaseModel):
    embedding: FaceEmbedding
    rect: Optional[List[int]]
    image: Optional[str]


class FacesResponse(BaseModel):
    status: Optional[str] = ""
    reply: List[FaceElement]


async def do_post(session, data, url, headers):
    async with session.post(url, data=data, headers=headers) as response:
        return await response.text()


@app.post("/images", response_model=ImagesResponse)
async def find_images(payload: RequestImages):
    async with aiohttp.ClientSession() as session:
        headers = {"content-type": "application/json"}
        tasks = []
        async for _, img_b64 in tools.aiter(payload.data):
            img_req = RequestDetect(value=img_b64).json()
            task = do_post(session, img_req, IMAGE_URL, headers)
            tasks.append(task)

        start_time = time.time()
        faces = await asyncio.gather(*tasks)
        print('RequestDetect', time.time() - start_time)

    request = RequestEmbeddings(distance_threshold=payload.distance_threshold,
                                limit=payload.limit)
    counts = [0]
    for text in faces:
        try:
            resp = FacesResponse.parse_raw(text)
            counts.append(counts[-1] + len(resp.reply))
        except ValidationError:
            logging.error(traceback.format_exc())
            counts.append(counts[-1])
            continue

        for face in resp.reply:
            request.data.append(face.embedding.data)

    start_time = time.time()
    embeddings = await find_embeddings(request)
    print('RequestEmbeddings', time.time() - start_time)

    respone = ImagesResponse()
    if embeddings['detail']:
        raise HTTPException(status_code=500, detail=embeddings['detail'])

    for left, right in zip(counts[:-1], counts[1:]):
        img_faces = embeddings['result'][left:right]
        if len(img_faces) == 0:
            img_faces = [SearchResult(detail='face detection ValidationError')]
        respone.result.append(img_faces)

    return respone


@app.post("/embeddings", response_model=EmbeddingsResponse)
async def find_embeddings(payload: RequestEmbeddings):
    nrows = len(payload.data)
    try:
        payload.data = tools.bytes_concat(map(base64.b64decode, payload.data))
    except:
        logging.error(traceback.format_exc())
        response = EmbeddingsResponse(detail='incorrect base64 encoding')
        return response

    data = payload.dict()
    data['nrows'] = nrows
    response = tcpclient.send(HOST, PORT, data, MessageType.FRONTEND)
    if response['detail']:
        raise HTTPException(status_code=500, detail=response['detail'])

    return response