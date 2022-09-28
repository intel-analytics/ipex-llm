
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

    local_ip = os.environ["MASTER_ADDR"]

    process = subprocess.Popen(cmd,
                               env={"WORLD_SIZE":f"{args.nnodes}", "RANK":"0", "MASTER_ADDR":local_ip, "MASTER_PORT":"12345"},
                               stdout=subprocess.PIPE)

    # config.load_incluster_config()
    config.load_incluster_config()

    v1 = client.CoreV1Api()

    for i in range(args.nnodes - 1):
        metadata = client.V1ObjectMeta(name=f'nano-run-worker-{i}')
        envs = [
            client.V1EnvVar(name="WORLD_SIZE", value=f"{args.nnodes}"),
            client.V1EnvVar(name="RANK", value=f"{i + 1}"),
            client.V1EnvVar(name="MASTER_ADDR", value=local_ip),
            client.V1EnvVar(name="MASTER_PORT", value="12345")
        ]
        container = client.V1Container(name="pytorch",
                                       image=args.image,
                                       env=envs,
                                       command=cmd)

        pod_spec = client.V1PodSpec(containers=[container], restart_policy="Never")
        pod_body = client.V1Pod(metadata=metadata, spec=pod_spec, kind='Pod', api_version='v1')
        pod = v1.create_namespaced_pod(namespace="default", body=pod_body)
    
    process.wait()
    print(process.stdout.read().decode("utf-8"))

if __name__ == '__main__':
    main()