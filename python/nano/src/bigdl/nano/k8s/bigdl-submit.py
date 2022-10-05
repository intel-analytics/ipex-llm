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

    parser.add_argument(
        "--namespace",
        type=str,
        default="default"
    )

    parser.add_argument(
        '--env',
        nargs=2,
        action='append'
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

    #
    # Positional arguments.
    #

    parser.add_argument("run_command", nargs=REMAINDER)

    return parser

def parse_args():
    parser = get_args_parser()
    return parser.parse_args()

def create_master_pod(v1_api, namespace, app_id, command,
                      image, master_port, world_size, extra_envs,
                      pod_cpu, pod_memory):
    pod_name = f'bigdl-{app_id}-master'
    pod_labels = {"bigdl-app": app_id, "bigdl-app-type": "master"}
    metadata = client.V1ObjectMeta(name=pod_name,
                                   labels=pod_labels)
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

    for env in extra_envs:
        envs.append(
            client.V1EnvVar(name=env[0], value=env[1]),
        )
    resource = client.V1ResourceRequirements(limits={"cpu": pod_cpu,
                                                     "memory": pod_memory},
                                             requests={"cpu": pod_cpu,
                                                       "memory": pod_memory})
    container = client.V1Container(name="pytorch",
                                   image=image,
                                   env=envs,
                                   command=command,
                                   resources=resource)

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy="Never")
    pod_body = client.V1Pod(api_version='v1',
                            metadata=metadata,
                            kind='Pod',
                            spec=pod_spec)
    pod = v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
    print(f"Created Master Pod: {pod_name}")
    return pod_name, pod_labels

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

def create_worker_pods(v1_api, world_size, app_id,
                       master_service_name, master_service_port,
                       image, command, extra_envs,
                       pod_cpu, pod_memory):

    for i in range(world_size - 1):
        pod_name = f'bigdl-{app_id}-worker-{i + 1}'
        metadata = client.V1ObjectMeta(name=pod_name,
                                       labels={"bigdl-app": app_id,
                                               "bigdl-app-type": "worker"})
        envs = [
            client.V1EnvVar(name="WORLD_SIZE", value=f"{world_size}"),
            client.V1EnvVar(name="RANK", value=f"{i + 1}"),
            client.V1EnvVar(name="MASTER_ADDR", value=master_service_name),
            client.V1EnvVar(name="MASTER_PORT", value=master_service_port)
        ]

        for env in extra_envs:
            envs.append(
                client.V1EnvVar(name=env[0], value=env[1]),
            )
        resource = client.V1ResourceRequirements(limits={"cpu": pod_cpu,
                                                         "memory": pod_memory},
                                                 requests={"cpu": pod_cpu,
                                                           "memory": pod_memory})
        container = client.V1Container(name="pytorch",
                                       image=image,
                                       env=envs,
                                       command=command,
                                       resources=resource)

        pod_spec = client.V1PodSpec(containers=[container], restart_policy="Never")
        pod_body = client.V1Pod(metadata=metadata, spec=pod_spec, kind='Pod', api_version='v1')

        pod = v1_api.create_namespaced_pod(namespace="default", body=pod_body)
        print(f"Created Rank {i + 1} Pod: {pod_name}")

def main():

    args = parse_args()

    run_command = args.run_command

    command = ["python"] + run_command

    app_id = str(uuid4())[:7]

    config.load_config()

    v1 = client.CoreV1Api()

    master_pod_name, master_pod_labels = create_master_pod(v1_api=v1,
                                                           namespace=args.namespace,
                                                           app_id=app_id,
                                                           command=command,
                                                           image=args.image,
                                                           master_port=args.master_port,
                                                           world_size=args.nnodes,
                                                           extra_envs=args.env,
                                                           pod_cpu=args.pod_cpu,
                                                           pod_memory=args.pod_memory)
    service_name, service_port = create_master_service(v1_api=v1,
                                                       namespace=args.namespace,
                                                       master_pod_name=master_pod_name,
                                                       master_pod_labels=master_pod_labels,
                                                       master_port=args.master_port)
    create_worker_pods(v1_api=v1,
                       world_size=args.nnodes,
                       app_id=app_id,
                       master_service_name=service_name,
                       master_service_port=service_port,
                       image=args.image,
                       command=command,
                       extra_envs=args.env,
                       pod_cpu=args.pod_cpu,
                       pod_memory=args.pod_memory)

    print("You can use the following commands to check out the pods status and logs.")
    print(f"**** kubectl get pods -l bigdl-app={app_id} ****")
    print(f"**** kubectl logs {master_pod_name} ****")
    

if __name__ == '__main__':
    main()
