#!/usr/bin/python3
from bigdl.serving.client import InputQueue, OutputQueue
from bigdl.common.encryption_utils import encrypt_with_AES_GCM
import os
import cv2
import json
import time
from optparse import OptionParser
import base64
from multiprocessing import Process
import redis
import yaml
import argparse
from numpy import *

RESULT_PREFIX = "cluster-serving_"
name = "serving_stream"


def main(args):
    if args.image_num % args.proc_num != 0:
        raise EOFError("Please make sure that image push number can be divided by multi-process number")
    redis_args = {}
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    redis_url = config.get('redisUrl')
    if redis_url:
        host = redis_url.split(':')[0]
        port = redis_url.split(':')[1]
        redis_args = {'host': host, 'port': port}
    if config.get('redisSecureEnabled'):
        if not os.path.isdir(args.keydir):
            raise EOFError("Please set secure key path")
        redis_args['ssl'] = 'True'
        redis_args['ssl_cert_reqs'] = 'none'
        redis_args['ssl_certfile'] = redis_args['ssl_ca_certs'] = os.path.join(args.keydir, "server.crt")
        redis_args['ssl_keyfile'] = os.path.join(args.keydir, "server.key")
    encrypt = config.get('recordEncrypted')

    DB = redis.StrictRedis(**redis_args)
    redis_args.pop('ssl_cert_reqs', None)

    try:
        print("Entering initial dequeue")
        output_api = OutputQueue(**redis_args)
        start = time.time()
        res = output_api.dequeue()
        end = time.time()
        print("Dequeued", len(res), "records in", end - start, "sec, dequeue fps:", len(res) / (end - start))
        print("Initial dequeue completed")
    except Exception:
        print("Dequeue error encountered")

    e2e_start = image_enqueue(redis_args, args.image_num, args.proc_num, args.image_path, encrypt)
    e2e_end, dequeue_num, num_invalid = image_dequeue(DB, args.image_num)
    num_valid = maximum(dequeue_num - num_invalid, 0)
    duration = e2e_end - e2e_start
    print("Served", num_valid, "images in", duration, "sec, e2e throughput is", num_valid / duration,
          "images/sec, excluded", num_invalid, "invalid results")


def image_enqueue(redis_args, img_num, proc_num, path, encrypt):
    print("Entering enqueue")
    input_api = InputQueue(**redis_args)
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    data = cv2.imencode(".jpg", img)[1]
    img_encoded = base64.b64encode(data).decode("utf-8")
    if encrypt:
        img_encoded = encrypt_with_AES_GCM(img_encoded, "secret", "salt")
        print("Record encoded")

    img_per_proc = int(img_num / proc_num)
    procs = []

    def push_image(image_num, index, proc_id):
        print("Entering enqueue", proc_id)
        for i in range(image_num):
            input_api.enqueue("my-img-" + str(i + index), t={"b64": img_encoded})

    start = time.time()
    for i in range(proc_num):
        proc = Process(target=push_image, args=(img_per_proc, i * img_per_proc, i,))
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()

    end = time.time()
    print(img_num, "images enqueued")
    print("total enqueue time:", end - start)
    fps = img_num / (end - start)
    print("enqueue fps:", fps)

    return start


def image_dequeue(DB, img_num):
    print("Entering dequeue")
    dequeue_num = 0
    num_invalid = 0
    start = time.time()
    while dequeue_num < img_num:
        pipe = DB.pipeline()
        res_list = DB.keys(RESULT_PREFIX + name + ':*')
        for res in res_list:
            pipe.hgetall(res.decode('utf-8'))
        res_dict_list = pipe.execute()
        for res_dict in res_dict_list:
            try:
                res_val = res_dict[b'value'].decode('utf-8')
            except Exception:
                print("Irregular result dict:", res_dict)
                num_invalid += 1
                continue
            if res_val == 'NaN':
                num_invalid += 1
        num_res = len(res_list)
        if num_res > 0:
            dequeue_num += num_res
            print("Received", dequeue_num, "results, including", num_invalid, "invalid results")
            DB.delete(*res_list)

    print("Total dequeue time:", time.time() - start)
    return time.time(), dequeue_num, num_invalid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', help='path of cluster serving config.yaml', default='../../config.yaml')
    parser.add_argument('--image_path', '-i', help='path of test image', default='ILSVRC2012_val_00000001.JPEG')
    parser.add_argument('--image_num', '-n', type=int, help='number of iterations to push image', default=1000)
    parser.add_argument('--proc_num', '-p', type=int, help='number of procs', default=10)
    parser.add_argument('--keydir', '-k', help='key files directory path', default='../keys')
    args = parser.parse_args()
    main(args)
