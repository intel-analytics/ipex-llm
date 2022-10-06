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
from argparse import REMAINDER, ArgumentParser
import functools

def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

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
        "--master_port",
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


    #
    # Positional arguments.
    #

    parser.add_argument("main_script", type=str)

    parser.add_argument("main_script_args", nargs=REMAINDER)

    return parser


def parse_args():
    parser = get_args_parser()
    return parser.parse_args()


class FakeKubeResponse:
    def __init__(self, json_str):
        self.data = json_str


def deserialize_volume_object(json_str, api_client):
    res = FakeKubeResponse(json_str)
    return api_client.deserialize(res,'V1Volume')


def deserialize_volume_mounts_object(json_str, api_client):
    res = FakeKubeResponse(json_str)
    return api_client.deserialize(res,'V1VolumeMount')


def create_pod(pod_name,
               pod_labels,
               rank,
               world_size,
               master_addr,
               master_port,
               app_id,
               extra_envs,
               pod_cpu,
               pod_memory,
               image,
               command,
               volume_strs,
               volume_mount_strs):
    api_cliet = client.ApiClient()
    metadata = client.V1ObjectMeta(name=pod_name,
                                   labels=pod_labels)
    client.V1EnvVarSource()
    envs = [
        client.V1EnvVar(name="WORLD_SIZE", value=f"{world_size}"),
        client.V1EnvVar(name="RANK", value=rank),
        client.V1EnvVar(name="MASTER_ADDR", value=master_addr),
        client.V1EnvVar(name="MASTER_PORT", value=master_port), # random selection？
        client.V1EnvVar(name="APP_ID", value=app_id)
    ]

    for env in extra_envs:
        envs.append(
            client.V1EnvVar(name=env[0], value=env[1]),
        )
    resource = client.V1ResourceRequirements(limits={"cpu": pod_cpu,
                                                     "memory": pod_memory},
                                             requests={"cpu": pod_cpu,
                                                       "memory": pod_memory})
    volumn_mounts = [deserialize_volume_mounts_object(json_str, api_cliet)
                 for json_str in volume_mount_strs]
    container = client.V1Container(name="pytorch",
                                   image=image,
                                   env=envs,
                                   command=command,
                                   resources=resource,
                                   volume_mounts=volumn_mounts)

    volumes = [deserialize_volume_object(json_str, api_cliet) for json_str in volume_strs]

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy="Never",
                                volumes=volumes)
    pod_body = client.V1Pod(api_version='v1',
                            metadata=metadata,
                            kind='Pod',
                            spec=pod_spec)
    return pod_body


def create_master_service(v1_api, namespace,
                          master_pod_name, master_pod_labels, master_port):
    v1_api = client.CoreV1Api()
    service_name = f'{master_pod_name}-service'
    metadata = client.V1ObjectMeta(name=service_name)
    port = client.V1ServicePort(protocol="TCP",
                                port=int(master_port),
                                target_port=int(master_port))

    service_spec = client.V1ServiceSpec(selector=master_pod_labels,
                                        ports=[port])

    service = client.V1Service(api_version="v1",
                               kind="Service",
                               metadata=metadata,
                               spec=service_spec)

    service = v1_api.create_namespaced_service(namespace=namespace, body=service)

    print(f"Created Master Service: {service_name}")
    return service_name, master_port


def create_master_pod(v1_api, namespace, pod_name, pod_labels, create_pod_fn):
    pod_body = create_pod_fn(rank="0", pod_name=pod_name, pod_labels=pod_labels)
    pod = v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
    print(f"Created Master Pod: {pod_name}")


def create_worker_pods(v1_api, namespace, world_size, app_id, create_pod_fn):

    for i in range(world_size - 1):
        pod_name = f'bigdl-{app_id}-worker-{i + 1}'
        pod_labels = {"bigdl-app": app_id,
                      "bigdl-app-type": "worker"}
        pod_body = create_pod_fn(rank=str(i + 1), pod_name=pod_name, pod_labels=pod_labels)
        pod = v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
        print(f"Created Rank {i + 1} Pod: {pod_name}")


def main():

    args = parse_args()

    command = ["python", args.main_script] + args.main_script_args

    app_id = str(uuid4())[:7]

    config.load_config()

    v1 = client.CoreV1Api()

    master_pod_name = f'bigdl-{app_id}-master'
    master_pod_labels = {"bigdl-app": app_id, "bigdl-app-type": "master"}

    service_name, service_port = create_master_service(v1_api=v1,
                                                       namespace=args.namespace,
                                                       master_pod_name=master_pod_name,
                                                       master_pod_labels=master_pod_labels,
                                                       master_port=args.master_port)


    create_pod_fn = functools.partial(create_pod,
                                      world_size=args.nnodes,
                                      master_addr=service_name,
                                      master_port=service_port,
                                      app_id=app_id,
                                      extra_envs=args.env,
                                      pod_cpu=args.pod_cpu,
                                      pod_memory=args.pod_memory,
                                      image=args.image,
                                      command=command,
                                      volume_strs=args.volume,
                                      volume_mount_strs=args.volume_mount)

    create_master_pod(v1_api=v1,
                      namespace=args.namespace,
                      pod_name=master_pod_name,
                      pod_labels=master_pod_labels,
                      create_pod_fn=create_pod_fn)

    create_worker_pods(v1_api=v1,
                       namespace=args.namespace,
                       world_size=args.nnodes,
                       app_id=app_id,
                       create_pod_fn=create_pod_fn)

    print("You can use the following commands to check out the pods status and logs.")
    print(f"**** kubectl get pods -l bigdl-app={app_id} ****")
    print(f"**** kubectl logs {master_pod_name} ****")
    

# if __name__ == '__main__':
#     main()
