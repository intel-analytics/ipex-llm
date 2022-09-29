
import subprocess
import sys
import os
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
        default="10.239.45.10/arda/nano-run",
        help="Rank of the node for multi-node distributed training.",
    )

    #
    # Positional arguments.
    #

    parser.add_argument("cmd", nargs=REMAINDER)

    return parser


def parse_args():
    parser = get_args_parser()
    return parser.parse_args()

def main():

    args = parse_args()

    cmd = args.cmd

    cmd = [sys.executable] + cmd[1:]

    master_ip = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    

    process = subprocess.Popen(cmd,
                               env={"WORLD_SIZE": f"{args.nnodes}",
                                    "RANK": "0",
                                    "MASTER_ADDR": master_ip,
                                    "MASTER_PORT": master_port})

    config.load_incluster_config()

    v1 = client.CoreV1Api()

    app_id = os.environ["APP_ID"]

    for i in range(args.nnodes - 1):
        pod_name = f'bigdl-{app_id}-worker-{i + 1}'
        metadata = client.V1ObjectMeta(name=pod_name,
                                       labels={"bigdl-app": app_id,
                                               "bigdl-app-type": "worker"})
        envs = [
            client.V1EnvVar(name="WORLD_SIZE", value=f"{args.nnodes}"),
            client.V1EnvVar(name="RANK", value=f"{i + 1}"),
            client.V1EnvVar(name="MASTER_ADDR", value=master_ip),
            client.V1EnvVar(name="MASTER_PORT", value=master_port)
        ]
        container = client.V1Container(name="pytorch",
                                       image=args.image,
                                       env=envs,
                                       command=cmd)

        pod_spec = client.V1PodSpec(containers=[container], restart_policy="Never")
        pod_body = client.V1Pod(metadata=metadata, spec=pod_spec, kind='Pod', api_version='v1')
        pod = v1.create_namespaced_pod(namespace="default", body=pod_body)
        print(f"Created Rank {i + 1} Pod: {pod_name}")
    
    process.wait()

if __name__ == '__main__':
    main()