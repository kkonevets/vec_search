import socket
import struct
from enum import Enum
import logging

SIZE_OF_LENGTH = struct.calcsize('>L')


class MessageType(Enum):
    FRONTEND = 0
    BATCH = 1


def send_msg(sock, msg):
    # Prefix each message with a 8-byte length (network byte order)
    msg = struct.pack('>L', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, SIZE_OF_LENGTH)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>L', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, toread):
    buf = bytearray(toread)
    view = memoryview(buf)
    while view:
        nbytes = sock.recv_into(view)
        if not nbytes:
            logging.error('recv ERROR %d %d', nbytes, toread)
            return None
        view = view[nbytes:]  # slicing views is cheap
    return buf