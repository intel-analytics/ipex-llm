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

from uuid import uuid4
from kubernetes import client, config
from kubernetes.client import ApiClient
from argparse import REMAINDER, ArgumentParser
import functools
from typing import Dict, List, Callable, Optional
import yaml
import json
from os import path
from bigdl.nano.utils.common import invalidInputError


def _get_args_parser() -> ArgumentParser:

    parser = ArgumentParser(description="BigDL Training Launcher")

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
        "--driver_port",
        type=str,
        default="12345",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="default"
    )

    parser.add_argument(
        '--env',
        nargs=2,
        action='append',
        default=[],
        help="pass environment variable to be set in all pods, "
             "such as --env http_proxy http://host:port"
    )

    parser.add_argument(
        '--pod_cpu',
        type=str,
        default="2",
    )

    parser.add_argument(
        '--pod_memory',
        type=str,
        default="1G"
    )

    parser.add_argument(
        '--pod_epc_memory',
        type=str,
        default="34359738368",
        help="The EPC memory allocated to the container (in bytes) if SGX mode is enabled"
    )

    parser.add_argument(
        '--volume',
        type=str,
        action='append',
        default=[],
        help="A Json string defining one volume in all pods"
    )

    parser.add_argument(
        '--volume_mount',
        type=str,
        action='append',
        default=[],
        help='A Json string specifying one volumeMount in all containers'
    )

    parser.add_argument(
        '--submit_pod_template',
        action='store_true',
        default=False,
        help='If set, indicate the main_script is a pod template yaml file'
    )

    parser.add_argument(
        '--use_command',
        action="store_true",
        default=False,
        help="If set, the script will use the command line arguments as pod's entrypoint"
    )

    parser.add_argument(
        '--sgx_enabled',
        action="store_true",
        default=False,
        help="If set, the corresponding sgx-related device-plugin arguments will be added"
    )

    parser.add_argument(
        '--node_label',
        nargs=2,
        action='append',
        default=[],
        help="choose which node to run with labels"
    )

    #
    # Positional arguments.
    #

    parser.add_argument("main_script", type=str)

    parser.add_argument("main_script_args", nargs=REMAINDER)

    return parser


def _parse_args():
    parser = _get_args_parser()
    return parser.parse_args()


class _FakeKubeResponse:
    def __init__(self, json_str: str):
        self.data = json_str


def _deserialize_volume_object(json_str: str, api_client: ApiClient) -> object:
    res = _FakeKubeResponse(json_str)
    return api_client.deserialize(res, 'V1Volume')


def _deserialize_volume_mounts_object(json_str: str, api_client: ApiClient) -> object:
    res = _FakeKubeResponse(json_str)
    return api_client.deserialize(res, 'V1VolumeMount')


def _deserialize_pod_object(json_str: str, api_client: ApiClient) -> object:
    res = _FakeKubeResponse(json_str)
    return api_client.deserialize(res, 'V1Pod')


def _get_json_str_from_yaml_file(file_name: str) -> Optional[str]:
    with open(path.abspath(file_name)) as f:
        yml_document_all = yaml.safe_load_all(f)
        for obj in yml_document_all:
            return json.dumps(obj)

    invalidInputError(False, "submitted yaml file is empty")
    return None


def _create_pod(pod_name: str,
                pod_labels: Dict[str, str],
                rank: str,
                world_size: int,
                driver_addr: str,
                driver_port: str,
                app_id: str,
                extra_envs: List[List[str]],
                labels: List[List[str]],
                pod_cpu: str,
                pod_memory: str,
                pod_epc_memory: str,
                image: str,
                command: str,
                use_command: bool,
                sgx_enabled: bool,
                volume_strs: List[str],
                volume_mount_strs: List[str],
                pod_file_template_str: Optional[str]) -> client.V1Pod:
    api_client = client.ApiClient()
    metadata = client.V1ObjectMeta(name=pod_name,
                                   labels=pod_labels)
    envs = [
        client.V1EnvVar(name="WORLD_SIZE", value=f"{world_size}"),
        client.V1EnvVar(name="RANK", value=rank),
        client.V1EnvVar(name="MASTER_ADDR", value=driver_addr),
        client.V1EnvVar(name="MASTER_PORT", value=driver_port),
        client.V1EnvVar(name="APP_ID", value=app_id)
    ]

    if pod_file_template_str is not None:
        pod_body: client.V1Pod = _deserialize_pod_object(pod_file_template_str,
                                                         api_client)
        pod_body.metadata.name = pod_name
        pod_body.metadata.labels.update(pod_labels)
        pod_body.spec.containers[0].env.extend(envs)
        pod_body.spec.restart_policy = "Never"
    else:
        for env in extra_envs:
            envs.append(
                client.V1EnvVar(name=env[0], value=env[1]),
            )

        node_selector = {}
        for label in labels:
            node_selector[label[0]] = label[1]

        requests = {
            "cpu": pod_cpu,
            "memory": pod_memory,
        }

        limits = {
            "cpu": pod_cpu,
            "memory": pod_memory,
        }
        if sgx_enabled:
            requests["sgx.intel.com/epc"] = pod_epc_memory
            requests["sgx.intel.com/enclave"] = "1"
            requests["sgx.intel.com/provision"] = "1"
            limits["sgx.intel.com/enclave"] = "1"
            limits["sgx.intel.com/provision"] = "1"
            limits["sgx.intel.com/epc"] = pod_epc_memory
        resource = client.V1ResourceRequirements(limits=limits,
                                                 requests=requests)
        volume_mounts = [_deserialize_volume_mounts_object(json_str, api_client)
                         for json_str in volume_mount_strs]
        if use_command:
            container = client.V1Container(name="pytorch",
                                           image=image,
                                           env=envs,
                                           command=command,
                                           resources=resource,
                                           volume_mounts=volume_mounts)
        else:
            container = client.V1Container(name="pytorch",
                                           image=image,
                                           env=envs,
                                           args=command,
                                           resources=resource,
                                           volume_mounts=volume_mounts)

        volumes = [_deserialize_volume_object(json_str, api_client) for json_str in volume_strs]

        pod_spec = client.V1PodSpec(containers=[container],
                                    restart_policy="Never",
                                    volumes=volumes,
                                    node_selector=node_selector)
        pod_body = client.V1Pod(api_version='v1',
                                metadata=metadata,
                                kind='Pod',
                                spec=pod_spec)

    return pod_body


def _create_driver_service(v1_api: client.CoreApi,
                           namespace: str,
                           driver_pod_name: str,
                           driver_pod_labels: str,
                           driver_port: str):
    service_name = f'{driver_pod_name}-service'
    metadata = client.V1ObjectMeta(name=service_name)
    port = client.V1ServicePort(protocol="TCP",
                                port=int(driver_port),
                                target_port=int(driver_port))

    service_spec = client.V1ServiceSpec(selector=driver_pod_labels,
                                        ports=[port])

    service = client.V1Service(api_version="v1",
                               kind="Service",
                               metadata=metadata,
                               spec=service_spec)

    service = v1_api.create_namespaced_service(namespace=namespace, body=service)

    print(f"Created Driver Service: {service_name}")
    return service_name, driver_port


def _create_driver_pod(v1_api: client.CoreApi,
                       namespace: str,
                       pod_name: str,
                       pod_labels: Dict[str, str],
                       create_pod_fn: Callable):
    pod_body = create_pod_fn(rank="0", pod_name=pod_name, pod_labels=pod_labels)
    pod = v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
    print(f"Created Driver Pod: {pod_name}")


def _create_worker_pods(v1_api: client.CoreApi,
                        namespace: str,
                        world_size: int,
                        app_id: str,
                        create_pod_fn: Callable):

    for i in range(world_size - 1):
        pod_name = f'bigdl-{app_id}-worker-{i + 1}'
        pod_labels = {"bigdl-app": app_id,
                      "bigdl-app-type": "worker"}
        pod_body = create_pod_fn(rank=str(i + 1), pod_name=pod_name, pod_labels=pod_labels)
        pod = v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
        print(f"Created Rank {i + 1} Pod: {pod_name}")


def main():
    """Entry point of bigdl-submit command line tool."""
    args = _parse_args()

    app_id = str(uuid4())[:7]

    config.load_config()

    v1 = client.CoreV1Api()

    driver_pod_name = f'bigdl-{app_id}-driver'
    driver_pod_labels = {"bigdl-app": app_id, "bigdl-app-type": "driver"}

    service_name, service_port = _create_driver_service(v1_api=v1,
                                                        namespace=args.namespace,
                                                        driver_pod_name=driver_pod_name,
                                                        driver_pod_labels=driver_pod_labels,
                                                        driver_port=args.driver_port)
    if args.submit_pod_template:
        template_json_str = _get_json_str_from_yaml_file(args.main_script)
        command = None
    else:
        template_json_str = None
        command = ["python", args.main_script] + args.main_script_args

    create_pod_fn = functools.partial(_create_pod,
                                      world_size=args.nnodes,
                                      driver_addr=service_name,
                                      driver_port=service_port,
                                      app_id=app_id,
                                      extra_envs=args.env,
                                      labels=args.node_label,
                                      pod_cpu=args.pod_cpu,
                                      pod_memory=args.pod_memory,
                                      pod_epc_memory=args.pod_epc_memory,
                                      image=args.image,
                                      command=command,
                                      use_command=args.use_command,
                                      sgx_enabled=args.sgx_enabled,
                                      volume_strs=args.volume,
                                      volume_mount_strs=args.volume_mount,
                                      pod_file_template_str=template_json_str
                                      )

    _create_driver_pod(v1_api=v1,
                       namespace=args.namespace,
                       pod_name=driver_pod_name,
                       pod_labels=driver_pod_labels,
                       create_pod_fn=create_pod_fn)

    _create_worker_pods(v1_api=v1,
                        namespace=args.namespace,
                        world_size=args.nnodes,
                        app_id=app_id,
                        create_pod_fn=create_pod_fn)

    print("You can use the following commands to check out the pods status and logs.")
    print(f"**** kubectl get pods -l bigdl-app={app_id} ****")
    print(f"**** kubectl logs {driver_pod_name} ****")
