from messaging import send_msg, recv_msg, MessageType
import socket
import bson
from typing import Optional, Dict
import numpy as np

from tools import HOST, PORT


def send(ip,
         port,
         data: dict,
         msg_type: MessageType = None,
         and_response: bool = True) -> Optional[Dict]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        if msg_type:
            data['type'] = msg_type.value
        message = bson.BSON.encode(data)
        send_msg(sock, message)
        if and_response:
            response = recv_msg(sock)
            return bson.BSON.decode(response)


if __name__ == "__main__":
    v = np.random.rand(512).astype('f').tobytes()
    data = {'data': v, 'nrows': 1}
    # result = send(HOST, PORT, data, MessageType.FRONTEND)
    result = send(HOST, PORT, data, MessageType.BATCH, False)
    print(result)