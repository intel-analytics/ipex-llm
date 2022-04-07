import logging

import grpc
import recall_pb2
import recall_pb2_grpc
import os
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('172.168.3.114:8084') as channel:
        stub = recall_pb2_grpc.RecallStub(channel)
        id_list = [767, 903, 163, 168, 367, 269, 368, 449, 150]
        for id in id_list:
            request = recall_pb2.Query(userID=449, k=10)
            candidates = stub.searchCandidates(request)
            print(candidates)



if __name__ == '__main__':
    logging.basicConfig()
    run()