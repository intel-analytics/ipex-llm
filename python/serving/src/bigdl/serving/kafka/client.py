from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import kafka_errors
import json
from bigdl.serving.schema import *


RESULT_PREFIX = "cluster-serving_"

class InputQueue:
    def __init__(self, frontend_url=None, **kwargs):
        host = kwargs.get("host") if kwargs.get("host") else "localhost"
        port = kwargs.get("port") if kwargs.get("port") else "9092"
        self.topic_name = kwargs.get("topic_name") if kwargs.get("topic_name") else "serving_stream"
        self.interval_if_error = 1
        for key in ["host", "port", "topic_name"]:
            if key in kwargs:
                kwargs.pop(key)
        # create a kafka producer  
        self.db = KafkaProducer(bootstrap_servers=host+":"+port,
                                key_serializer=lambda k: json.dumps(k).encode('utf-8'),
                                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                **kwargs)
        
    def enqueue(self, uri, **data):
        b64str = self.data_to_b64(**data)
        d = {"key":uri, "value":{"uri":uri, "data":b64str}}        
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
    
    def __enqueue_data(self, data):
        # send a message {key:value}
        future = self.db.send(self.topic_name, **data)
        try:
            future.get(timeout=10) # check if send successfully
        except kafka_errors:  # throw kafka_errors if failed
            traceback.format_exc()
        print("Write to Kafka successful")
    
    @staticmethod
    def base64_encode_image(img):
        # base64 encode the input NumPy array
        return base64.b64encode(img).decode("utf-8")
    
    def close(self):
        self.db.close()


class OutputQueue:
    def __init__(self, host=None, port=None, group_id='group-1',
                 auto_offset_reset='earliest', **kwargs):
        host = host if host else "localhost"
        port = port if port else "9092"
        self.topic_name = kwargs.get("topic_name") if kwargs.get("topic_name") else RESULT_PREFIX+ "serving_stream"
        
        for key in ["host", "port", "topic_name"]:
            if key in kwargs:
                kwargs.pop(key)
        # create a kafka consumer    
        self.db = KafkaConsumer(self.topic_name, bootstrap_servers=host+":"+port, 
                                group_id=group_id, auto_offset_reset=auto_offset_reset, **kwargs)
        
    def dequeue(self):
        # poll get records
        records = self.db.poll(timeout_ms = 500)
        self.db.commit()
        decoded = {}
        for tp, messages in records.items():
                for message in messages:
                    res_id = message.key.decode()
                    res_value = message.value.decode()
                    decoded[res_id] = self.get_ndarray_from_b64(res_value)  # decode value
        return decoded
    
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
    
    def close(self):
        self.db.close()
