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

import subprocess
import sys
import os
from uuid import uuid4
from kubernetes import client, config
from argparse import REMAINDER, ArgumentParser

def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="Nano Training Launcher")

    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="yangw1234/bigdl-submit-demo:latest",
        help="Rank of the node for multi-node distributed training.",
    )

    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
    )

    parser.add_argument(
        "--service_account_name",
        type=str,
        default="default"
    )

    #
    # Positional arguments.
    #

    parser.add_argument("run_command", nargs=REMAINDER)

    return parser

def parse_args():
    parser = get_args_parser()
    return parser.parse_args()

def create_master_pod(app_id, command, image, master_port, world_size, service_account_name):
    config.load_config()
    v1 = client.CoreV1Api()
    pod_name = f'bigdl-{app_id}-master'
    metadata = client.V1ObjectMeta(name=pod_name,
                                   labels={"bigdl-app": app_id,
                                           "bigdl-app-type": "master"})
    client.V1EnvVarSource()
    envs = [
        client.V1EnvVar(name="WORLD_SIZE", value=f"{world_size}"),
        client.V1EnvVar(name="RANK", value=f"0"),
        client.V1EnvVar(name="MASTER_ADDR",
                        value_from=client.V1EnvVarSource(
                            field_ref=client.V1ObjectFieldSelector(field_path="status.podIP"))),
        client.V1EnvVar(name="MASTER_PORT", value=master_port), # random selection？
        client.V1EnvVar(name="APP_ID", value=app_id)
    ]
    container = client.V1Container(name="pytorch",
                                   image=image,
                                   env=envs,
                                   command=command)

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy="Never",
                                service_account_name=service_account_name)
    pod_body = client.V1Pod(api_version='v1',
                            metadata=metadata,
                            kind='Pod',
                            spec=pod_spec)
    pod = v1.create_namespaced_pod(namespace="default", body=pod_body)
    print(f"Created Master Pod: {pod_name}")

def main():

    args = parse_args()


    run_command = args.run_command

    command_prefix = ["python",
                      "/workspace/master_run.py",
                      "--nnodes",
                      str(args.nnodes),
                      "--image",
                      args.image,
                      ]

    app_id = str(uuid4())[:7]

    create_master_pod(app_id=app_id,
                      command=command_prefix + run_command,
                      image=args.image,
                      master_port=args.master_port,
                      world_size=args.nnodes,
                      service_account_name=args.service_account_name)

if __name__ == '__main__':
    main()
