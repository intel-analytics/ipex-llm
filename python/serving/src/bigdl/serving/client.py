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

import redis
import time
from bigdl.serving.schema import *
import httpx
import json
import uuid

RESULT_PREFIX = "cluster-serving_"


def http_json_to_ndarray(json_str):
    # currently there is no http user use batch predict, so batch is not implemented here
    # to add batch predict, replace 0 index to [0, batch_size)
    res_dict = json.loads(json.loads(json.loads(json_str)["predictions"][0])['value'])
    data, shape = res_dict['data'], res_dict['shape']
    array = np.array(data)
    array = array.reshape(shape)
    return array


def http_response_to_ndarray(response):
    if response.status_code == 200:
        response_str = response.text
        return http_json_to_ndarray(response_str)
    elif response.status_code == 400:
        print("Invalid input format, valid example:")
        print("""{
"instances": [
   {
     "tag": "foo",
     "signal": [1, 2, 3, 4, 5],
     "sensor": [[1, 2], [3, 4]]
   }
]
}
""")
    else:
        print("Error when calling Cluster Serving Http server, error code:", response.status_code)
    print("WARNING: Server returns invalid response, so you will get []")
    return "[]"


def perdict(frontend_url, request_str):
    httpx.post(frontend_url + "/predict", data=request_str)


class API:
    """
    base level of API control
    select data pipeline here, Redis/Kafka/...
    interface preserved for API class
    """
    def __init__(self, host=None, port=None, name="serving_stream"):
        self.name = name
        self.host = host if host else "localhost"
        self.port = port if port else "6379"

        self.db = redis.StrictRedis(host=self.host,
                                    port=self.port, db=0)
        try:
            self.db.xgroup_create(name, "serving")
        except Exception:
            print("redis group exist, will not create new one")


class InputQueue(API):
    def __init__(self, frontend_url=None, **kwargs):
        super().__init__(**kwargs)
        self.frontend_url = frontend_url
        if self.frontend_url:
            # frontend_url is provided, using frontend
            try:
                res = httpx.get(frontend_url)
                if res.status_code == 200:
                    httpx.PoolLimits(max_keepalive=1, max_connections=1)
                    self.cli = httpx.Client()
                    print("Attempt connecting to Cluster Serving frontend success")
                else:
                    raise ConnectionError()
            except Exception as e:
                print("Connection error, please check your HTTP server. Error msg is ", e)
        else:
            self.output_queue = OutputQueue(**kwargs)

        # TODO: these params can be read from config in future
        self.input_threshold = 0.6
        self.interval_if_error = 1

    def predict(self, request_data, timeout=5):
        """
        :param request_data:
        :param time_sleep:
        :return:
        """
        def json_to_ndarray_dict(json_str):
            ndarray_dict = {}
            data_dict = json.loads(json_str)['instances'][0]
            for key in data_dict.keys():
                ndarray_dict[key] = np.array(data_dict[key])
            return ndarray_dict

        if self.frontend_url:
            response = self.cli.post(self.frontend_url + "/predict", data=request_data)
            predictions = json.loads(response.text)['predictions']
            processed = predictions[0].lstrip("{value=").rstrip("}")
        else:
            try:
                json.loads(request_data)
                input_dict = json_to_ndarray_dict(request_data)
            except Exception as e:
                if isinstance(request_data, dict):
                    input_dict = request_data
                else:
                    input_dict = {'t': request_data}

            uri = str(uuid.uuid4())
            self.enqueue(uri, **input_dict)
            processed = "[]"
            time_sleep = 0.001
            while time_sleep < timeout:
                processed = self.output_queue.query_and_delete(uri)
                if processed != "[]":
                    break
                time.sleep(time_sleep)
                time_sleep += 0.001
        return processed

    def enqueue(self, uri, **data):
        b64str = self.data_to_b64(**data)
        d = {"uri": uri, "data": b64str}
        self.__enqueue_data(d)

    def data_to_b64(self, **data):
        sink = pa.BufferOutputStream()
        field_list = []
        data_list = []
        for key, value in data.items():
            field, data = get_field_and_data(key, value)
            field_list.append(field)
            data_list.append(data)

        schema = pa.schema(field_list)
        batch = pa.RecordBatch.from_arrays(
            data_list, schema)

        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        buf = sink.getvalue()
        b = buf.to_pybytes()
        b64str = self.base64_encode_image(b)
        return b64str

    def enqueue_tensor(self, uri, data):
        """
        deprecated
        """
        if isinstance(data, np.ndarray):
            # tensor
            data = [data]
        if not isinstance(data, list):
            raise Exception("Your input is invalid, only List of ndarray and ndarray are allowed.")

        sink = pa.BufferOutputStream()
        writer = None
        for d in data:
            shape = np.array(d.shape)
            d = d.astype("float32").flatten()

            data_field = pa.field("data", pa.list_(pa.float32()))
            shape_field = pa.field("shape", pa.list_(pa.int64()))
            tensor_type = pa.struct([data_field, shape_field])

            tensor = pa.array([{'data': d}, {'shape': shape}],
                              type=tensor_type)

            tensor_field = pa.field(uri, tensor_type)
            schema = pa.schema([tensor_field])

            batch = pa.RecordBatch.from_arrays(
                [tensor], schema)
            if writer is None:
                # initialize
                writer = pa.RecordBatchFileWriter(sink, batch.schema)
            writer.write_batch(batch)

        writer.close()
        buf = sink.getvalue()
        b = buf.to_pybytes()
        tensor_encoded = self.base64_encode_image(b)
        d = {"uri": uri, "data": tensor_encoded}
        self.__enqueue_data(d)

    def __enqueue_data(self, data):
        inf = self.db.info()
        try:
            if inf['used_memory'] >= inf['maxmemory'] * self.input_threshold\
                    and inf['maxmemory'] != 0:
                raise redis.exceptions.ConnectionError
            self.db.xadd(self.name, data)
            print("Write to Redis successful")
        except redis.exceptions.ConnectionError:
            print("Redis queue is full, please wait for inference "
                  "or delete the unprocessed records.")
            time.sleep(self.interval_if_error)

        except redis.exceptions.ResponseError as e:
            print(e, "Please check if Redis version > 5, "
                     "if yes, memory may be full, try dequeue or delete.")
            time.sleep(self.interval_if_error)

    @staticmethod
    def base64_encode_image(img):
        # base64 encode the input NumPy array
        return base64.b64encode(img).decode("utf-8")


class OutputQueue(API):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dequeue(self):
        res_list = self.db.keys(RESULT_PREFIX + self.name + ':*')
        decoded = {}
        for res in res_list:
            res_dict = self.db.hgetall(res.decode('utf-8'))
            res_id = res.decode('utf-8').split(":")[1]
            res_value = res_dict[b'value'].decode('utf-8')
            if res_value == "NaN":
                decoded[res_id] = "NaN"
            else:
                decoded[res_id] = self.get_ndarray_from_b64(res_value)
            self.db.delete(res)
        return decoded

    def query_and_delete(self, uri):
        return self.query(uri, True)

    def query(self, uri, delete=False):
        res_dict = self.db.hgetall(RESULT_PREFIX + self.name + ':' + uri)

        if not res_dict or len(res_dict) == 0:
            return "[]"
        if delete:
            self.db.delete(RESULT_PREFIX + self.name + ':' + uri)
        s = res_dict[b'value'].decode('utf-8')
        if s == "NaN":
            return s
        return self.get_ndarray_from_b64(s)

    def get_ndarray_from_b64(self, b64str):
        b = base64.b64decode(b64str)
        a = pa.BufferReader(b)
        c = a.read_buffer()
        myreader = pa.ipc.open_stream(c)
        r = [i for i in myreader]
        assert len(r) > 0
        if len(r) == 1:
            return self.get_ndarray_from_record_batch(r[0])
        else:
            l = []
            for ele in r:
                l.append(self.get_ndarray_from_record_batch(ele))
            return l

    def get_ndarray_from_record_batch(self, record_batch):
        data = record_batch[0].to_numpy()
        shape_list = record_batch[1].to_pylist()
        shape = [i for i in shape_list if i]
        ndarray = data.reshape(shape)
        return ndarray
