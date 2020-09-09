faiss server with separate gunicorn clients communicating through tcp socket

1) faiss_server process:
    - at startup loads vectors from logs and builds an index, creates a child process 2)
    - accepts data from gunicorn processes via socket, this happens via FIFO in the queue at the kernel level
    - makes search queries in one main thread so that faiss does not create more threads than cores
    - accepts a batch from a child process, slows down search queries and adds the batch to the index
2) child process faiss_server:
    - listens to rabbitmq and climbs into the hermitage for vectors and immediately writes them to the log, adds them to the batch, when the batch size reaches the threshold and gives the batch to the parent in time
    - the log is written in the form of two files:
        - text, in which each line is the id of the cut face, for example "face1 @ 123 @ vk.com"
        - face vectors themselves in binary format
        - when building the index, it is necessary to check that the number of lines in these 2 files coincide and that the last line in the binary file is not broken
3) gunicorn processes serving user requests, sending data over a socket to faiss_server