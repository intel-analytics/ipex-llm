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

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import kafka_errors
import traceback
import json
import sys


def producer_demo():
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        key_serializer=lambda k: json.dumps(k).encode(),
        value_serializer=lambda v: json.dumps(v).encode())
    # send three example messages
    # a topic that doesn't exist will be created
    for i in range(0, 3):
        future = producer.send(
            'serving_stream',
            key='test',  # same key will be sent to same partition
            value=str(i),
            partition=0)  # send to partition 0
        print("send {}".format(str(i)))
        try:
            future.get(timeout=10)  # check if send successfully
        except kafka_errors:  # throw kafka_errors if failed
            traceback.format_exc()
    producer.close()


def consumer_demo():
    consumer = KafkaConsumer(
        'cluster-serving_serving_stream',
        bootstrap_servers=['localhost:9092'],
    )
    for message in consumer:
        print("receive, key: {}, value: {}".format(
            json.loads(message.key.decode()),
            json.loads(message.value.decode())
        ))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
