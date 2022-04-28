#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import grpc
from friesian.example.serving.generated import recall_pb2_grpc, recall_pb2
import os
import pandas as pd
import argparse
import time
import threading


if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']


class Timer:
    def __init__(self):
        self.count = 0
        self.total = 0

    def reset(self):
        self.count = 0
        self.total = 0

    def add(self, time_elapse):
        self.total += time_elapse * 1000
        self.count += 1

    def get_stat(self):
        return self.count, self.total / self.count


class SimilarityClient():
    def __init__(self, stub):
        self.stub = stub

    def search(self, id, k):
        request = recall_pb2.Query(userID=id, k=k)
        try:
            candidates = self.stub.searchCandidates(request)
            return candidates.candidate
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))
            return

    def getMetrics(self):
        try:
            msg = self.stub.getMetrics(recall_pb2.ServerMessage())
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))
        logging.info("Got metrics: " + msg.str)

    def resetMetrics(self):
        try:
            self.stub.resetMetrics(recall_pb2.ServerMessage())
        except Exception as e:
            logging.warning("RPC failed:{}".format(e))


def single_thread_client(client, ids, k, timer):
    result_dict = dict()
    for id in ids:
        search_start = time.perf_counter()
        results = client.search(id, k)
        result_dict[id] = results
        print(id, ":", results)
        search_end = time.perf_counter()
        timer.add(search_end - search_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Friesian')
    parser.add_argument('--data_dir', type=str, default="item_ebd.parquet",
                        help='path of parquet files')
    parser.add_argument('--target', type=str, default=None,
                        help='The target of recall service url')
    logging.basicConfig(filename="client.log", level=logging.INFO)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_dir)
    id_list = df["tweet_id"].unique()
    n_thread = 4

    with grpc.insecure_channel(args.target) as channel:
        stub = recall_pb2_grpc.RecallStub(channel)
        client = SimilarityClient(stub)

        client_timer = Timer()
        thread_list = []
        size = len(id_list) // n_thread
        for ith in range(n_thread):
            ids = id_list[ith * size: (ith + 1) * size]
            thread = threading.Thread(target=single_thread_client,
                                      args=(client, ids, 10, client_timer))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        count, avg = client_timer.get_stat()
        client_timer.reset()
        client.getMetrics()
        client.resetMetrics()
