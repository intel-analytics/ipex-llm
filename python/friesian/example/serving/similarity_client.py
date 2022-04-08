import logging
import grpc
import recall_pb2
import recall_pb2_grpc
import os
import pandas as pd
import argparse

if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

class SimilarityClient():
    def __init__(self, stub):
        self.stub = stub

    def search(self, id, k):
        request = recall_pb2.Query(userID=id, k=k)
        try:
            results = self.stub.searchCandidates(request)
            return results
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))

    def getMetrics(self):
        try:
            results = self.stub.getMetrics()
            return results
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))

    def resetMetrics(self):
        try:
            self.stub.resetMetrics()
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Friesian')
    parser.add_argument('--data_dir', type=str, default="item_ebd.parquet",
                        help='path of parquet files')
    parser.add_argument('--target', type=str, default=None,
                        help='The target of recall service url')
    logging.basicConfig()
    args = parser.parse_args()

    df = pd.read_parquet(args.data_dir)
    id_list = df["tweet_id"].unique()[0: 100]

    with grpc.insecure_channel(args.target) as channel:
        stub = recall_pb2_grpc.RecallStub(channel)
        client = SimilarityClient(stub)
        for id in id_list:
            candidates = client.search(id, 10)
            print(candidates)
        client.getMetrics()
        client.resetMetrics()

